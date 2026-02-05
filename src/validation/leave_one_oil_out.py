"""
Leave-one-oil-out cross-validation (enhanced distance-based analysis).

This validation method tests the model's ability to recognize when presented
with a completely novel oil type. Unlike classification, this uses distance
metrics in the LDA-transformed space.

Enhanced analysis:
- Compares left-out oil distances to training class boundaries
- Calculates intra-class dispersion (class "radius")
- Visualizes each left-out oil scenario
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST, before any imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.validation.validation import (
    load_data, save_results, get_group_ids_for_oil_types
)


def resolve_output_path(output_dir: str) -> Path:
    """Resolve output directory path relative to project root."""
    if not Path(output_dir).is_absolute():
        return project_root / output_dir
    return Path(output_dir)


def calculate_class_boundary(distances):
    """
    Calculate class boundary as mean + 2*std (95% of samples within boundary).

    Args:
        distances: Array of distances from samples to their class center

    Returns:
        boundary: The boundary distance (mean + 2*std)
    """
    return np.mean(distances) + 2 * np.std(distances)


def plot_single_leave_out_scenario(left_out_class, X_train_lda, y_train, X_test_lda,
                                    class_centers, class_boundaries, output_path):
    """
    Create visualization for a single leave-one-oil-out scenario.

    Shows the training classes as clusters with their boundaries,
    and where the left-out oil falls relative to them.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # For each pair of LDA dimensions
    n_dims = X_train_lda.shape[1]
    plot_idx = 0

    for i in range(min(n_dims, 12)):
        for j in range(i + 1, min(n_dims, 12)):
            if plot_idx >= 12:
                break

            ax = axes[plot_idx]

            # Plot training classes
            unique_classes = np.unique(y_train)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

            for cls, color in zip(unique_classes, colors):
                mask = y_train == cls
                ax.scatter(X_train_lda[mask, i], X_train_lda[mask, j],
                          c=[color], label=f'Train {cls}', alpha=0.5, s=50)

                # Plot class center
                center = class_centers[str(cls)]
                ax.scatter(center[i], center[j], c=color, marker='*',
                          s=200, edgecolors='black', linewidths=1.5)

                # Plot class boundary (circle)
                boundary = class_boundaries[str(cls)]
                circle = plt.Circle((center[i], center[j]), boundary,
                                   color=color, fill=False, linestyle='--',
                                   alpha=0.5)
                ax.add_patch(circle)

            # Plot left-out oil
            ax.scatter(X_test_lda[:, i], X_test_lda[:, j],
                      c='red', marker='^', s=150,
                      edgecolors='black', linewidths=2,
                      label=f'Left-out {left_out_class}', zorder=10)

            ax.set_xlabel(f'LDA Component {i+1}')
            ax.set_ylabel(f'LDA Component {j+1}')
            ax.set_title(f'LDA {i+1} vs {j+1}')
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)

            plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, 12):
        axes[idx].axis('off')

    plt.suptitle(f'Leave-One-Oil-Out Analysis: Oil Type {left_out_class} Left Out\n'
                 f'Training classes shown with boundaries (μ+2σ)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_enhanced_distance_distribution(all_results, output_path):
    """
    Create comprehensive plot showing separation ratios for all left-out oils.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data
    oil_labels = sorted(all_results.keys())
    min_distances = []
    nearest_boundaries = []
    separation_ratios = []

    for oil in oil_labels:
        result = all_results[oil]
        min_distances.append(result['min_distance_to_nearest_class'])
        nearest_boundaries.append(result['nearest_class_boundary'])
        separation_ratios.append(result['separation_ratio'])

    # Panel 1: Bar chart of separation ratios
    colors = ['green' if r > 1.5 else 'orange' if r > 1.0 else 'red' for r in separation_ratios]
    bars = ax1.bar(range(len(oil_labels)), separation_ratios, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Left-Out Oil Type', fontsize=12)
    ax1.set_ylabel('Separation Ratio (Min Distance / Class Boundary)', fontsize=12)
    ax1.set_title('Enhanced Leave-One-Oil-Out: Separation Ratios\n'
                  '(Ratio > 1 means outside training class boundary)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(oil_labels)))
    ax1.set_xticklabels(oil_labels)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Threshold (1.0)')
    ax1.axhline(y=1.5, color='blue', linestyle=':', linewidth=2, label='Excellent (1.5)')
    ax1.axhline(y=np.mean(separation_ratios), color='green', linestyle='-',
                linewidth=1, label=f'Mean: {np.mean(separation_ratios):.2f}')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(separation_ratios) * 1.1])

    # Add value labels on bars
    for i, (bar, ratio) in enumerate(zip(bars, separation_ratios)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}', ha='center', va='bottom', fontsize=9)

    # Panel 2: Scatter plot of distance vs boundary
    ax2.scatter(nearest_boundaries, min_distances, c=colors, s=100,
                alpha=0.7, edgecolors='black', linewidth=1)
    ax2.plot([0, max(nearest_boundaries)], [0, max(nearest_boundaries)],
             'k--', linewidth=1, label='Ratio = 1 (on boundary)')
    ax2.set_xlabel('Nearest Training Class Boundary (μ+2σ)', fontsize=12)
    ax2.set_ylabel('Distance from Left-Out Oil to Nearest Class', fontsize=12)
    ax2.set_title('Distance vs Class Boundary\n(Points above line are well-separated)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Add oil labels to scatter plot
    for i, oil in enumerate(oil_labels):
        ax2.annotate(oil, (nearest_boundaries[i], min_distances[i]),
                    fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_leave_one_oil_out(data_path="data/raw/data-923.csv",
                          output_dir="experiments/leave_one_oil_out"):
    """
    Run enhanced leave-one-oil-out validation.

    Strategy:
    - For each of 13 oil types, train on the other 12
    - Transform data to LDA space
    - Calculate class centers and boundaries (μ + 2σ)
    - Calculate distances from left-out oil to training classes
    - Calculate separation ratio = (min distance) / (nearest class boundary)
    - Create individual visualizations for each scenario
    """
    print("="*60)
    print("LEAVE-ONE-OIL-OUT (ENHANCED DISTANCE-BASED ANALYSIS)")
    print("="*60)

    # Load data
    X, y = load_data(data_path)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(set(y))} oil types")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    # Get group IDs for oil types (all 6 samples of each oil form a group)
    oil_groups = get_group_ids_for_oil_types(y)

    print(f"\nConfiguration:")
    print(f"  Strategy: Leave one oil type completely out")
    print(f"  Groups: 13 (one per oil type)")
    print(f"  Samples per group: 6 (replicates)")

    # Create LeaveOneGroupOut splitter
    logo = LeaveOneGroupOut()

    # Storage for results
    all_results = {}

    print(f"\nRunning leave-one-oil-out validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y_encoded, groups=oil_groups)):
        # Get the left-out class
        left_out_class_encoded = y_encoded[test_idx[0]]
        left_out_class = le.inverse_transform([left_out_class_encoded])[0]
        train_classes = le.inverse_transform(y_encoded[train_idx])

        print(f"\n  Fold {fold_idx + 1}/13: Leaving out oil type '{left_out_class}'")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Preprocess: StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Dimensionality reduction: LDA
        # Number of components = min(n_features, n_classes - 1)
        n_classes_train = len(np.unique(y_train))
        n_components = min(X_train.shape[1], n_classes_train - 1)

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_train_lda = lda.fit_transform(X_train_scaled, y_train)
        X_test_lda = lda.transform(X_test_scaled)

        # Calculate class centers and boundaries for each training class
        class_centers = {}
        class_boundaries = {}
        class_internal_distances = {}

        for cls in np.unique(y_train):
            cls_mask = y_train == cls
            cls_samples = X_train_lda[cls_mask]

            # Calculate class center
            center = np.mean(cls_samples, axis=0)
            class_centers[str(cls)] = center

            # Calculate distances from each sample to class center
            distances = np.linalg.norm(cls_samples - center, axis=1)
            class_internal_distances[str(cls)] = distances

            # Calculate class boundary (mean + 2*std)
            boundary = calculate_class_boundary(distances)
            class_boundaries[str(cls)] = boundary

        # Calculate distances from left-out oil to each training class
        distances_to_training_classes = {}

        for cls in np.unique(y_train):
            center = class_centers[str(cls)]
            boundary = class_boundaries[str(cls)]

            # Distance from left-out samples to this class center
            distances = np.linalg.norm(X_test_lda - center, axis=1)
            min_distance = float(np.min(distances))
            mean_distance = float(np.mean(distances))
            std_distance = float(np.std(distances))

            # Separation ratio: how far is left-out oil relative to class boundary
            separation_ratio = min_distance / boundary if boundary > 0 else float('inf')

            distances_to_training_classes[str(cls)] = {
                'class_name': str(le.inverse_transform([cls])[0]),
                'min_distance': min_distance,
                'mean_distance': mean_distance,
                'std_distance': std_distance,
                'class_center': center.tolist(),
                'class_boundary': boundary,
                'separation_ratio': separation_ratio
            }

        # Find nearest training class
        nearest_cls = min(distances_to_training_classes.items(),
                        key=lambda item: item[1]['min_distance'])[0]
        nearest_result = distances_to_training_classes[nearest_cls]

        # Store results for this fold
        all_results[left_out_class] = {
            'left_out_class': left_out_class,
            'left_out_samples': int(len(X_test)),
            'training_classes': [str(c) for c in np.unique(y_train)],
            'distances_to_training_classes': distances_to_training_classes,
            'min_distance_to_nearest_class': nearest_result['min_distance'],
            'nearest_class_name': nearest_result['class_name'],
            'nearest_class_boundary': nearest_result['class_boundary'],
            'separation_ratio': nearest_result['separation_ratio']
        }

        print(f"    Nearest class: {nearest_result['class_name']}")
        print(f"    Min distance: {nearest_result['min_distance']:.2f}")
        print(f"    Class boundary: {nearest_result['class_boundary']:.2f}")
        print(f"    Separation ratio: {nearest_result['separation_ratio']:.2f}")

        # Create individual visualization for this scenario
        output_path = resolve_output_path(output_dir)
        scenario_path = output_path / f'leave_out_{left_out_class}_analysis.png'
        plot_single_leave_out_scenario(
            left_out_class, X_train_lda, y_train, X_test_lda,
            class_centers, class_boundaries, str(scenario_path)
        )

    # Calculate summary statistics
    all_min_distances = [r['min_distance_to_nearest_class'] for r in all_results.values()]
    all_boundaries = [r['nearest_class_boundary'] for r in all_results.values()]
    all_ratios = [r['separation_ratio'] for r in all_results.values()]

    mean_min_distance = np.mean(all_min_distances)
    mean_boundary = np.mean(all_boundaries)
    mean_ratio = np.mean(all_ratios)
    std_ratio = np.std(all_ratios, ddof=1)

    well_separated_count = sum(1 for r in all_ratios if r > 1.0)
    excellent_count = sum(1 for r in all_ratios if r > 1.5)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nOverall Statistics:")
    print(f"  Mean of minimum distances: {mean_min_distance:.2f}")
    print(f"  Mean class boundary: {mean_boundary:.2f}")
    print(f"  Mean separation ratio: {mean_ratio:.2f} ± {std_ratio:.2f}")
    print(f"\nSeparation Quality:")
    print(f"  Well separated (ratio > 1): {well_separated_count}/{len(all_ratios)} ({well_separated_count/len(all_ratios)*100:.1f}%)")
    print(f"  Excellent (ratio > 1.5): {excellent_count}/{len(all_ratios)} ({excellent_count/len(all_ratios)*100:.1f}%)")

    print(f"\nDetailed Results by Left-Out Oil:")
    print(f"{'Oil':<6} {'Min Dist':>10} {'Boundary':>10} {'Ratio':>8} {'Status':>12}")
    print("-" * 56)
    for oil in sorted(all_results.keys()):
        r = all_results[oil]
        ratio = r['separation_ratio']
        if ratio > 1.5:
            status = "✓ Excellent"
        elif ratio > 1.0:
            status = "✓ Good"
        elif ratio > 0.8:
            status = "⚠ Moderate"
        else:
            status = "✗ Poor"
        print(f"{oil:<6} {r['min_distance_to_nearest_class']:>10.2f} {r['nearest_class_boundary']:>10.2f} {ratio:>8.2f} {status:>12}")

    # Prepare results for saving
    results = {
        "method": "leave_one_oil_out_enhanced",
        "description": "Enhanced leave-one-oil-out validation with distance-based analysis and class boundary comparison",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(le.classes_),
        "configuration": {
            "validation_strategy": "LeaveOneGroupOut",
            "n_groups": len(np.unique(oil_groups)),
            "samples_per_group": 6,
            "analysis_type": "distance-based with separation ratio"
        },
        "distance_analysis": {
            "by_left_out_oil": all_results,
            "summary_statistics": {
                "mean_of_min_distances": float(mean_min_distance),
                "mean_class_boundary": float(mean_boundary),
                "mean_separation_ratio": float(mean_ratio),
                "std_separation_ratio": float(std_ratio),
                "all_separation_ratios": [float(r) for r in all_ratios],
                "all_min_distances": [float(d) for d in all_min_distances],
                "all_boundaries": [float(b) for b in all_boundaries],
                "well_separated_count": well_separated_count,
                "excellent_count": excellent_count,
                "total_count": len(all_ratios)
            }
        }
    }

    # Save results
    save_results(results, output_dir)

    # Create enhanced distance distribution plot
    output_path = resolve_output_path(output_dir)
    plot_path = output_path / 'enhanced_distance_analysis.png'
    plot_enhanced_distance_distribution(all_results, str(plot_path))
    print(f"\nEnhanced distance analysis plot saved to {plot_path}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if well_separated_count == len(all_ratios):
        print("✓ All left-out oils are outside training class boundaries (ratio > 1)")
        print("  Model can clearly identify novel samples")
    if excellent_count >= len(all_ratios) * 0.8:
        print(f"✓ {excellent_count}/{len(all_ratios)} oils show excellent separation (ratio > 1.5)")
        print(f"  Mean separation ratio of {mean_ratio:.2f} indicates strong separability")

    print(f"\nKey finding:")
    print(f"  Left-out oils are {mean_ratio:.1f}x further from training classes than")
    print(f"  the classes' own boundaries (μ+2σ), indicating good class separation.")

    return results


if __name__ == "__main__":
    run_leave_one_oil_out()
