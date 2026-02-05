"""
Validation framework for oil type discrimination model.

This package contains comprehensive validation methods to address
overfitting concerns:

- leave_replicate_out: Leave-replicate-out cross-validation
- multi_seed_cv: Multi-random-seed cross-validation with confidence intervals
- feature_ablation: Feature ablation (learning curve) analysis
- leave_one_oil_out: Leave-one-oil-out distance-based analysis
- label_permutation_test: Label permutation test for statistical significance
- generate_validation_report: Generate comprehensive validation report
- run_all_validations: Run all validation methods at once
"""

from .validation import (
    load_data,
    create_model,
    create_pipeline,
    save_results,
    get_group_ids_for_replicates,
    get_group_ids_for_oil_types,
    encode_labels,
    calculate_confidence_interval
)

__all__ = [
    'load_data',
    'create_model',
    'create_pipeline',
    'save_results',
    'get_group_ids_for_replicates',
    'get_group_ids_for_oil_types',
    'encode_labels',
    'calculate_confidence_interval'
]
