"""
Multi-random-seed cross-validation with confidence intervals.

This validation method demonstrates model stability by reporting performance
across multiple random seeds with confidence intervals.

Expected: Narrow 95% CI indicates stable performance that generalizes well.
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
    load_data, create_pipeline, save_results, calculate_confidence_interval
)


def resolve_output_path(output_dir: str) -> Path:
    """Resolve output directory path relative to project root."""
    if not Path(output_dir).is_absolute():
        return project_root / output_dir
    return Path(output_dir)


def plot_stability_boxplot(scores_by_seed, output_path):
    """Create boxplot showing score distribution across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot
    bp = ax.boxplot([scores_by_seed], tick_labels=['All Seeds'],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red',
                                  markeredgecolor='red', markersize=8))

    # Color the box
    bp['boxes'][0].set_facecolor('lightblue')

    # Add individual points
    x_jitter = np.random.normal(1, 0.05, len(scores_by_seed))
    ax.scatter(x_jitter, scores_by_seed, alpha=0.5, s=50, c='darkblue', edgecolors='black', linewidth=0.5, zorder=3)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Stability Across 20 Random Seeds\n(Stratified 2-Fold CV)', fontsize=14, pad=20)
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    mean = np.mean(scores_by_seed)
    std = np.std(scores_by_seed)
    ci_lower, ci_upper = calculate_confidence_interval(scores_by_seed)[1:]

    stats_text = f'Mean: {mean:.4f}\nStd: {std:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_multi_seed_cv(data_path="data/raw/data-923.csv",
                      n_seeds=20,
                      n_splits=2,
                      output_dir="experiments/multi_seed"):
    """
    Run cross-validation with multiple random seeds.

    Strategy:
    - For each of 20 random seeds, run stratified 2-fold CV
    - Report mean accuracy, std, and 95% confidence interval
    """
    print("="*60)
    print("MULTI-RANDOM-SEED CROSS-VALIDATION")
    print("="*60)

    # Load data
    X, y = load_data(data_path)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(set(y))} oil types")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    print(f"\nConfiguration:")
    print(f"  Random seeds: {n_seeds}")
    print(f"  CV folds per seed: {n_splits} (stratified)")
    print(f"  Total evaluations: {n_seeds * n_splits}")

    # Run CV for each seed
    print(f"\nRunning cross-validation for {n_seeds} seeds...")
    scores_by_seed = []
    scores_by_fold = []

    for seed in range(n_seeds):
        if (seed + 1) % 5 == 0 or seed == 0:
            print(f"  Seed {seed + 1}/{n_seeds}...")

        # Create model
        pipe = create_pipeline()

        # Stratified k-fold with this seed
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # Cross-validation
        fold_scores = cross_val_score(pipe, X, y_encoded, cv=skf, scoring="accuracy")
        mean_score = np.mean(fold_scores)

        scores_by_seed.append(mean_score)
        scores_by_fold.extend(fold_scores)

    scores_by_seed = np.array(scores_by_seed)
    scores_by_fold = np.array(scores_by_fold)

    # Calculate statistics
    mean, ci_lower, ci_upper = calculate_confidence_interval(scores_by_seed)
    std = np.std(scores_by_seed, ddof=1)
    min_val = np.min(scores_by_seed)
    max_val = np.max(scores_by_seed)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nMean Accuracy: {mean:.4f} ({mean*100:.2f}%)")
    print(f"Std Deviation: {std:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Min: {min_val:.4f} ({min_val*100:.2f}%)")
    print(f"Max: {max_val:.4f} ({max_val*100:.2f}%)")
    print(f"Range: {max_val - min_val:.4f}")

    # Calculate CI width as percentage of mean
    ci_width = ci_upper - ci_lower
    ci_width_percent = (ci_width / mean * 100) if mean > 0 else 0
    print(f"CI Width: {ci_width:.4f} ({ci_width_percent:.2f}% of mean)")

    # Prepare results for saving
    results = {
        "method": "multi_seed_cv",
        "description": f"Cross-validation with {n_seeds} random seeds to assess stability",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(le.classes_),
        "configuration": {
            "n_seeds": n_seeds,
            "n_splits_per_seed": n_splits,
            "total_evaluations": n_seeds * n_splits
        },
        "metrics": {
            "mean_accuracy": float(mean),
            "mean_accuracy_percent": float(mean * 100),
            "std_deviation": float(std),
            "min_accuracy": float(min_val),
            "max_accuracy": float(max_val),
            "range": float(max_val - min_val),
            "confidence_interval_95": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "width": float(ci_width),
                "width_percent_of_mean": float(ci_width_percent)
            }
        },
        "raw_scores": {
            "by_seed_mean": scores_by_seed.tolist(),
            "all_folds": scores_by_fold.tolist()
        }
    }

    # Save results
    save_results(results, output_dir)

    # Save boxplot
    plot_path = resolve_output_path(output_dir) / "stability_boxplot.png"
    plot_stability_boxplot(scores_by_seed, str(plot_path))
    print(f"\nStability boxplot saved to {plot_path}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if ci_width_percent < 5:
        print("✓ Very narrow CI (<5% of mean) indicates excellent stability")
        print("  Model performance is consistent across different train/test splits")
    elif ci_width_percent < 10:
        print("✓ Narrow CI (5-10% of mean) indicates good stability")
        print("  Model performance is reasonably consistent")
    else:
        print("⚠ Wider CI suggests variability in performance")
        print("  Consider more data or regularization")

    return results


if __name__ == "__main__":
    run_multi_seed_cv()
