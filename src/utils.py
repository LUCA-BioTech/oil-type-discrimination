"""
Unified utility module for oil type discrimination project.

Provides shared functions for data loading, model creation, and result saving
to be used across all scripts (training, evaluation, validation, prediction).
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

import sys
# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from src.ml_model import MLModel


# =============================================================================
# Data Loading
# =============================================================================

def load_data(path: str = "data/raw/data-923.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the oil type discrimination dataset.

    Args:
        path: Path to the CSV file containing the data (relative to project root)

    Returns:
        X: Feature matrix of shape (78, 15) - 78 samples, 15 enzyme features
        y: Label array of shape (78,) - 13 oil types (A-M), 6 replicates each
    """
    # Resolve path relative to project root
    if not Path(path).is_absolute():
        full_path = project_root / path
    else:
        full_path = Path(path)

    df = pd.read_csv(full_path)
    X = df[[f"en{i}" for i in range(1, 16)]].values
    y = df["cate"].values
    return X, y


def load_data_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from any CSV file with the same column structure.

    This is used for prediction on new data files.

    Args:
        csv_path: Path to CSV file (can be absolute or relative to project root)

    Returns:
        X: Feature matrix
        y: Dummy label array (not used for prediction)
    """
    if not Path(csv_path).is_absolute():
        full_path = project_root / csv_path
    else:
        full_path = Path(csv_path)

    df = pd.read_csv(full_path)
    X = df[[f"en{i}" for i in range(1, 16)]].values
    # Return dummy labels for prediction (won't be used)
    y = np.zeros(X.shape[0])
    return X, y


def scale_data(X: np.ndarray) -> np.ndarray:
    """
    Scale data using StandardScaler.

    Args:
        X: Feature matrix

    Returns:
        Scaled feature matrix
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# =============================================================================
# Model Creation
# =============================================================================

def create_model() -> MLModel:
    """
    Create a fresh instance of the ML model.

    Returns:
        MLModel instance with untrained pipeline
    """
    return MLModel()


def create_pipeline() -> Pipeline:
    """
    Create a sklearn-compatible pipeline for cross-validation.

    This is useful for cross_val_score which requires a sklearn estimator.
    The MLModel class wraps this pipeline internally.

    Returns:
        Pipeline with scaler, LDA, and Naive Bayes
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
        ("nb", GaussianNB())
    ])
    return pipe


# =============================================================================
# Results Saving
# =============================================================================

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save validation results to JSON file.

    Args:
        results: Dictionary containing metrics and metadata
        output_dir: Directory to save results (relative to project root, creates if doesn't exist)
    """
    # Resolve path relative to project root
    if not Path(output_dir).is_absolute():
        full_output_dir = project_root / output_dir
    else:
        full_output_dir = Path(output_dir)

    os.makedirs(full_output_dir, exist_ok=True)
    output_path = full_output_dir / "metrics.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {output_path}")


# =============================================================================
# Label Utilities
# =============================================================================

def encode_labels(y: np.ndarray, encoder: LabelEncoder = None) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to integers.

    Args:
        y: Label array (strings like 'A', 'B', ...)
        encoder: Optional pre-fitted encoder

    Returns:
        Tuple of (encoded labels, fitted encoder)
    """
    if encoder is None:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y.astype(str))
    else:
        y_encoded = encoder.transform(y.astype(str))

    return y_encoded, encoder


# =============================================================================
# Statistical Utilities
# =============================================================================

def calculate_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate mean and confidence interval for a set of values.

    Args:
        values: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    n = len(values)

    # Z-score for 95% confidence
    z_score = 1.96 if confidence == 0.95 else 2.576

    margin_of_error = z_score * std / np.sqrt(n) if n > 0 else 0.0

    return mean, mean - margin_of_error, mean + margin_of_error


# =============================================================================
# Group ID Helpers for Cross-Validation
# =============================================================================

def get_group_ids_for_replicates(y: np.ndarray) -> np.ndarray:
    """
    Create group IDs for leave-replicate-out cross-validation.

    Groups samples by replicate number (0-5) across all oil types.
    This allows leaving out one complete replicate set (13 samples: 1 from each oil type).

    The data is ordered as: AAAAAA BBBBBB CCCCCC ... (6 consecutive samples per class)
    So replicate 0 = indices 0, 6, 12, 18, ... (1st sample of each class)
    And replicate 5 = indices 5, 11, 17, 23, ... (6th sample of each class)

    Args:
        y: Label array

    Returns:
        Array of group IDs, one per sample (values 0-5 for the 6 replicates)
    """
    group_ids = []
    for i in range(len(y)):
        # For sample at index i:
        # Class index = i // 6 (which class it belongs to: 0 for A, 1 for B, etc.)
        # Replicate index = i % 6 (which replicate within that class: 0-5)
        replicate_id = i % 6
        group_ids.append(replicate_id)

    return np.array(group_ids)


def get_group_ids_for_oil_types(y: np.ndarray) -> np.ndarray:
    """
    Create group IDs for leave-one-oil-out cross-validation.

    All 6 replicates of an oil type share the same group ID.
    This allows us to leave out entire oil types.

    Args:
        y: Label array

    Returns:
        Array of group IDs, one per sample
    """
    unique_classes = sorted(set(y))
    group_mapping = {cls: i for i, cls in enumerate(unique_classes)}
    return np.array([group_mapping[cls] for cls in y])
