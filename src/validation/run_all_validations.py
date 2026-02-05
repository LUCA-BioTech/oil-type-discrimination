#!/usr/bin/env python3
"""
Run all validation experiments and generate comprehensive report.

This script executes all validation methods in sequence to address
overfitting concerns for the oil type discrimination model.
"""

import sys
from pathlib import Path

# Add project root to path FIRST, before any imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.validation.leave_replicate_out import run_leave_replicate_out
from src.validation.multi_seed_cv import run_multi_seed_cv
from src.validation.feature_ablation import run_feature_ablation
from src.validation.leave_one_oil_out import run_leave_one_oil_out
from src.validation.label_permutation_test import run_label_permutation_test
from src.validation.generate_validation_report import main as generate_report


def main():
    """Run all validation experiments."""
    print("="*70)
    print(" "*15 + "COMPREHENSIVE VALIDATION SUITE")
    print("="*70)
    print("\nThis will run 5 validation methods to address overfitting concerns:")
    print("  1. Leave-Replicate-Out Cross-Validation")
    print("  2. Multi-Random-Seed Cross-Validation")
    print("  3. Feature Ablation (Learning Curve)")
    print("  4. Leave-One-Oil-Out (Distance-Based)")
    print("  5. Label Permutation Test")
    print("\nThen generate a comprehensive validation report.")
    print("="*70)

    # Stage 1: Leave-replicate-out
    print("\n" + "="*70)
    print("STAGE 1: Leave-Replicate-Out Cross-Validation")
    print("="*70)
    run_leave_replicate_out()

    # Stage 2: Multi-seed CV
    print("\n" + "="*70)
    print("STAGE 2: Multi-Random-Seed Cross-Validation")
    print("="*70)
    run_multi_seed_cv()

    # Stage 3: Feature ablation
    print("\n" + "="*70)
    print("STAGE 3: Feature Ablation (Learning Curve)")
    print("="*70)
    run_feature_ablation()

    # Stage 4: Leave-one-oil-out
    print("\n" + "="*70)
    print("STAGE 4: Leave-One-Oil-Out (Distance-Based)")
    print("="*70)
    run_leave_one_oil_out()

    # Stage 5: Label permutation test
    print("\n" + "="*70)
    print("STAGE 5: Label Permutation Test")
    print("="*70)
    run_label_permutation_test()

    # Stage 6: Generate comprehensive report
    print("\n" + "="*70)
    print("STAGE 6: Generate Comprehensive Validation Report")
    print("="*70)
    generate_report()

    print("\n" + "="*70)
    print(" "*20 + "ALL VALIDATIONS COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  - experiments/leave_replicate_out/")
    print("  - experiments/multi_seed/")
    print("  - experiments/feature_ablation/")
    print("  - experiments/leave_one_oil_out/")
    print("  - experiments/label_permutation/")
    print("  - experiments/validation_report/")
    print("\nKey files:")
    print("  - experiments/validation_report/validation_report.md")
    print("  - experiments/validation_report/validation_summary.png")


if __name__ == "__main__":
    main()
