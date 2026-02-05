"""
Label Permutation Test (标签置换检验).

This validation method tests whether the model learns meaningful feature-label
relationships rather than exploiting random patterns.
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


def plot_permutation_test(true_accuracy, shuffled_accuracies, p_value, output_path):
    """Create histogram showing null distribution and true accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of shuffled accuracies
    n, bins, patches = ax.hist(shuffled_accuracies, bins=30, alpha=0.6,
                                color='lightcoral', edgecolor='black', linewidth=0.5)

    # Plot true accuracy as vertical line
    ax.axvline(true_accuracy, color='darkblue', linewidth=3,
               label=f'True Accuracy: {true_accuracy:.4f}', linestyle='-')

    # Add significance markers
    ax.axvline(np.percentile(shuffled_accuracies, 95), color='red',
               linestyle='--', linewidth=1.5, alpha=0.7, label='95th percentile')

    # Add p-value text
    p_text = f'p-value: {p_value:.2e}' if p_value < 0.001 else f'p-value: {p_value:.4f}'
    ax.text(0.98, 0.95, p_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Label Permutation Test: Null Distribution vs True Accuracy\\n'
                 f'N={len(shuffled_accuracies)} permutations', fontsize=14, pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_label_permutation_test(data_path="data/raw/data-923.csv",
                                n_permutations=100,
                                n_splits=2,
                                output_dir="experiments/label_permutation"):
    """
    Run label permutation test.

    Args:
        data_path: Path to data file
        n_permutations: Number of permutations to run
        n_splits: Number of CV folds
        output_dir: Directory to save results
    """
    print("="*60)
    print("LABEL PERMUTATION TEST")
    print("="*60)

    # Load data
    X, y = load_data(data_path)
    print(f"\\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(set(y))} oil types")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    print(f"\\nConfiguration:")
    print(f"  Permutations: {n_permutations}")
    print(f"  CV folds: {n_splits} (stratified)")
    print(f"  Random seed for shuffling: 42 (reproducible)")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Step 1: Compute true accuracy
    print("\\nComputing true accuracy with original labels...")
    pipe = create_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    true_scores = cross_val_score(pipe, X, y_encoded, cv=cv)
    true_accuracy = np.mean(true_scores)
    print(f"  True accuracy: {true_accuracy:.4f} ({true_accuracy*100:.2f}%)")

    # Step 2: Compute null distribution from shuffled labels
    print(f"\\nRunning {n_permutations} permutations...")
    shuffled_accuracies = []

    for i in range(n_permutations):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        # Shuffle labels (break feature-label relationship)
        y_shuffled = np.random.permutation(y_encoded)

        # Compute accuracy with shuffled labels
        pipe = create_pipeline()
        shuffled_scores = cross_val_score(pipe, X, y_shuffled, cv=cv, scoring="accuracy")
        shuffled_accuracy = np.mean(shuffled_scores)
        shuffled_accuracies.append(shuffled_accuracy)

    shuffled_accuracies = np.array(shuffled_accuracies)

    # Step 3: Calculate statistics
    shuffled_mean = np.mean(shuffled_accuracies)
    shuffled_std = np.std(shuffled_accuracies, ddof=1)
    shuffled_min = np.min(shuffled_accuracies)
    shuffled_max = np.max(shuffled_accuracies)
    percentile_95 = np.percentile(shuffled_accuracies, 95)
    percentile_99 = np.percentile(shuffled_accuracies, 99)

    # Calculate p-value: proportion of shuffled accuracies >= true accuracy
    p_value = np.sum(shuffled_accuracies >= true_accuracy) / n_permutations

    # Print results
    print("\\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\\nTrue Accuracy: {true_accuracy:.4f} ({true_accuracy*100:.2f}%)")
    print(f"\\nNull Distribution (shuffled labels):")
    print(f"  Mean: {shuffled_mean:.4f} ({shuffled_mean*100:.2f}%)")
    print(f"  Std Dev: {shuffled_std:.4f}")
    print(f"  Min: {shuffled_min:.4f} ({shuffled_min*100:.2f}%)")
    print(f"  Max: {shuffled_max:.4f} ({shuffled_max*100:.2f}%)")
    print(f"  95th percentile: {percentile_95:.4f} ({percentile_95*100:.2f}%)")

    print(f"\\nStatistical Significance:")
    if p_value < 0.001:
        print(f"  p-value: {p_value:.2e}")
    else:
        print(f"  p-value: {p_value:.4f}")

    # Prepare results for saving
    results = {
        "method": "label_permutation_test",
        "description": "Label permutation test: compares true accuracy to null distribution from shuffled labels",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(le.classes_),
        "configuration": {
            "n_permutations": n_permutations,
            "n_splits": n_splits,
            "random_seed": 42
        },
        "true_accuracy": {
            "value": float(true_accuracy),
            "percent": float(true_accuracy * 100),
            "fold_scores": [float(x) for x in true_scores]
        },
        "null_distribution": {
            "mean": float(shuffled_mean),
            "std": float(shuffled_std),
            "min": float(shuffled_min),
            "max": float(shuffled_max),
            "percentile_95": float(percentile_95),
            "percentile_99": float(percentile_99),
            "all_accuracies": [float(x) for x in shuffled_accuracies]
        },
        "statistical_test": {
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "alpha": 0.05,
            "interpretation": "Extremely significant" if p_value < 0.001 else "Significant" if p_value < 0.01 else "Not significant"
        }
    }

    # Save results
    save_results(results, output_dir)

    # Save permutation test plot
    plot_path = resolve_output_path(output_dir) / "permutation_test.png"
    plot_permutation_test(true_accuracy, shuffled_accuracies, p_value, str(plot_path))
    print(f"\\nPermutation test plot saved to {plot_path}")

    print("\\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if p_value < 0.001:
        print("✓ EXTREMELY significant (p < 0.001)")
        print("  The model learns highly significant feature-label relationships")
    elif p_value < 0.01:
        print("✓ Highly significant (p < 0.01)")
        print("  The model learns significant feature-label relationships")
    elif p_value < 0.05:
        print("✓ Significant (p < 0.05)")
        print("  The model learns meaningful feature-label relationships")
    else:
        print("⚠ Not significant (p >= 0.05)")
        print("  Cannot reject null hypothesis")

    print(f"\\nKey finding:")
    print(f"  True accuracy ({true_accuracy*100:.1f}%) is {true_accuracy/shuffled_mean:.2f}x higher than")
    print(f"  the null distribution mean ({shuffled_mean*100:.1f}%)")

    return results


if __name__ == "__main__":
    run_label_permutation_test()
