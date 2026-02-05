"""
Validation framework for oil type discrimination model.

This module now imports utility functions from src/utils for consistency
across the entire codebase. Only validation-specific helpers remain here.
"""

from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

import sys
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import all utility functions from src/utils
from src.utils import (
    load_data,
    create_model,
    create_pipeline,
    save_results,
    encode_labels,
    calculate_confidence_interval,
    get_group_ids_for_replicates,
    get_group_ids_for_oil_types,
)


# Re-export for backward compatibility with existing validation scripts
__all__ = [
    'load_data',
    'create_model',
    'create_pipeline',
    'save_results',
    'encode_labels',
    'calculate_confidence_interval',
    'get_group_ids_for_replicates',
    'get_group_ids_for_oil_types',
    'resolve_output_path',
]


def resolve_output_path(output_dir: str) -> Path:
    """
    Resolve output directory path relative to project root.

    This is a validation-specific helper used by multiple validation scripts.

    Args:
        output_dir: Directory path (can be relative or absolute)

    Returns:
        Resolved absolute Path object
    """
    if not Path(output_dir).is_absolute():
        return project_root / output_dir
    return Path(output_dir)
