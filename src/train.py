"""
Training script for oil type discrimination model.

Uses stratified 5-fold cross-validation to evaluate model performance.
Results are saved to experiments/base_model/metrics.json
"""

import json
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import unified utilities
from src.utils import load_data, create_pipeline, save_results


def main():
    """Run 5-fold stratified cross-validation and save results."""
    print("="*60)
    print("TRAINING: 5-Fold Stratified Cross-Validation")
    print("="*60)

    # Load data using unified utility
    X, y = load_data("data/raw/data-923.csv")
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(set(y))} oil types")

    # Encode labels for sklearn CV
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    # Create pipeline using unified utility
    pipe = create_pipeline()

    # Configure cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Run cross-validation
    print("\nRunning 5-fold stratified cross-validation...")
    scores = cross_val_score(pipe, X, y_encoded, cv=cv, scoring="accuracy")

    # Prepare results
    results = {
        "method": "stratified_5fold_cv",
        "description": "5-fold stratified cross-validation with random_state=42",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": len(le.classes_),
        "cv_folds": 5,
        "cv_scores": scores.tolist(),
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "min_accuracy": float(np.min(scores)),
        "max_accuracy": float(np.max(scores))
    }

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nCross-validation scores: {scores}")
    print(f"Mean accuracy: {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)")
    print(f"Std deviation: {results['std_accuracy']:.4f}")
    print(f"Min accuracy: {results['min_accuracy']:.4f}")
    print(f"Max accuracy: {results['max_accuracy']:.4f}")

    # Save results using unified utility
    save_results(results, "experiments/base_model")


if __name__ == "__main__":
    main()
