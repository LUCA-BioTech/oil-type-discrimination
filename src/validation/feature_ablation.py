"""
Feature Ablation (Learning Curve) analysis.

This validation method tests discriminative power of different enzyme subsets,
demonstrating the extensibility of the sensor array system.

Expected: Accuracy increases with more enzymes, showing the value of
the sensor array approach.
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST, before any imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.validation.validation import (
    load_data, create_pipeline, save_results
)


def resolve_output_path(output_dir: str) -> Path:
    """Resolve output directory path relative to project root."""
    if not Path(output_dir).is_absolute():
        return project_root / output_dir
    return Path(output_dir)


def plot_learning_curve(enzyme_results, output_path):
    """Create learning curve plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    n_enzymes = [r["n_enzymes"] for r in enzyme_results]
    means = [r["mean"] for r in enzyme_results]
    stds = [r["std"] for r in enzyme_results]

    # Plot line with error bars
    ax.errorbar(n_enzymes, means, yerr=stds, marker="o", linestyle="-",
                linewidth=2, markersize=8, capsize=5, capthick=2,
                color="steelblue", ecolor="darkblue", label="Accuracy ± SD")

    # Add value labels
    for n, m in zip(n_enzymes, means):
        ax.text(n, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)

    ax.set_xlabel("Number of Enzymes in Sensor Array", fontsize=12)
    ax.set_ylabel("Accuracy (Stratified 2-Fold CV)", fontsize=12)
    ax.set_title("Feature Ablation: Learning Curve\nDiscriminative Power vs Array Size", fontsize=14, pad=20)
    ax.set_xticks(n_enzymes)
    ax.set_xticklabels([f"{n}" for n in n_enzymes])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    # Set y-axis limits with some padding
    ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_feature_ablation(data_path="data/raw/data-923.csv",
                         output_dir="experiments/feature_ablation"):
    """
    Run feature ablation study with different enzyme subsets.

    Strategy:
    - Test different enzyme subset sizes: 1, 3, 5, 10, 15 (all)
    - For each subset size, run stratified 2-fold CV
    - Report accuracy vs number of enzymes
    """
    print("="*60)
    print("FEATURE ABLATION (LEARNING CURVE)")
    print("="*60)

    # Load data
    X, y = load_data(data_path)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(set(y))} oil types")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    # Define enzyme subsets to test
    # Using evenly distributed enzymes for each subset size
    enzyme_subsets = {
        "1_enzyme": [0],                      # en1 only
        "3_enzymes": [0, 7, 14],              # en1, en8, en15 (evenly spaced)
        "5_enzymes": [0, 3, 7, 11, 14],       # en1, en4, en8, en12, en15
        "10_enzymes": list(range(10)),        # en1-en10
        "15_enzymes": list(range(15)),        # all enzymes (en1-en15)
    }

    print(f"\nTesting {len(enzyme_subsets)} enzyme subset configurations:")
    for name, indices in enzyme_subsets.items():
        print(f"  {name}: enzyme indices {indices}")

    print("\nRunning cross-validation for each subset...")

    results_by_subset = {}
    enzyme_results = []

    for name, indices in enzyme_subsets.items():
        n_enzymes = len(indices)
        print(f"\n  Testing {n_enzymes} enzyme(s)...")

        # Select subset
        X_subset = X[:, indices]

        # Create model
        pipe = create_pipeline()

        # Cross-validation (2-fold stratified for consistency)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X_subset, y_encoded, cv=cv, scoring="accuracy")

        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)

        print(f"    Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")

        results_by_subset[name] = {
            "n_enzymes": n_enzymes,
            "enzyme_indices": indices,
            "mean_accuracy": float(mean_score),
            "std_accuracy": float(std_score),
            "fold_scores": scores.tolist()
        }

        enzyme_results.append({
            "name": name,
            "n_enzymes": n_enzymes,
            "mean": mean_score,
            "std": std_score
        })

    # Print results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print("\nEnzyme Subset | Accuracy | Std Dev | Improvement")
    print("-" * 60)

    baseline = enzyme_results[0]["mean"]
    for r in enzyme_results:
        improvement = (r["mean"] - baseline) * 100
        n_enz = r["n_enzymes"]
        mean_val = r["mean"]
        std_val = r["std"]
        print(f"{n_enz:>14} | {mean_val:>.4f}   | {std_val:>.4f}  | +{improvement:.2f}%")

    # Prepare results for saving
    results = {
        "method": "feature_ablation",
        "description": "Feature ablation study testing discriminative power of different enzyme subset sizes",
        "n_samples": X.shape[0],
        "n_features_total": X.shape[1],
        "n_classes": len(le.classes_),
        "configuration": {
            "cv_strategy": "StratifiedKFold(n_splits=2, shuffle=True, random_state=42)",
            "n_subsets_tested": len(enzyme_subsets)
        },
        "results_by_subset": results_by_subset,
        "summary": {
            "min_accuracy": float(min(r["mean"] for r in enzyme_results)),
            "max_accuracy": float(max(r["mean"] for r in enzyme_results)),
            "improvement_min_to_max": float(max(r["mean"] for r in enzyme_results) - min(r["mean"] for r in enzyme_results))
        }
    }

    # Save results
    save_results(results, output_dir)

    # Save learning curve plot
    plot_path = resolve_output_path(output_dir) / "learning_curve.png"
    plot_learning_curve(enzyme_results, str(plot_path))
    print(f"\nLearning curve saved to {plot_path}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    improvement = results["summary"]["improvement_min_to_max"]
    if improvement > 0.2:
        print("✓ Significant improvement (>20%) with more enzymes")
        print("  Sensor array approach provides clear value")
    elif improvement > 0.1:
        print("✓ Moderate improvement (10-20%) with more enzymes")
        print("  Sensor array adds discriminative power")
    elif improvement > 0.05:
        print("✓ Some improvement (5-10%) with more enzymes")
        print("  Additional enzymes contribute marginally")
    else:
        print("⚠ Minimal improvement with more enzymes")
        print("  Consider feature selection or different enzyme combinations")

    results_summary_max_accuracy = results["summary"]["max_accuracy"]
    print(f"\nKey finding: {results_summary_max_accuracy:.1%} accuracy with full 15-enzyme array")

    return results


if __name__ == "__main__":
    run_feature_ablation()
