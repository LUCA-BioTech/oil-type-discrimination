"""
Leave-replicate-out cross-validation.

This validation method proves the model learns oil type reaction patterns
rather than memorizing individual samples. Each fold leaves out exactly
one replicate (sample) from the dataset.

Expected: High accuracy (>90%) indicates the model generalizes across
measurement variations within each oil type.
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST, before any imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

from src.validation.validation import (
    load_data, create_model, save_results, get_group_ids_for_replicates
)


def plot_confusion_matrix(cm, classes, output_path):
    """Create and save confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Proportion'})

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Leave-Replicate-Out Confusion Matrix\n(Each row normalized to sum to 1)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_leave_replicate_out(data_path="data/raw/data-923.csv",
                            output_dir="experiments/leave_replicate_out"):
    """
    Run leave-replicate-out cross-validation.

    Strategy:
    - 78 samples total (13 oil types × 6 replicates each)
    - Leave One Group Out: each fold leaves out 1 complete replicate set
    - Each replicate set = 13 samples (1 from each oil type)
    - Train on 65 samples (5 replicates × 13 oils), test on 13 samples (1 replicate × 13 oils)
    - Repeat 6 times (once per replicate set)
    """
    print("="*60)
    print("LEAVE-REPLICATE-OUT CROSS-VALIDATION")
    print("="*60)

    # Load data
    X, y = load_data(data_path)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(set(y))} oil types")
    print(f"Samples per class: 6 replicates")

    # Create group IDs (replicate number 0-5, same for all oil types)
    # This ensures each fold leaves out 1 complete replicate set
    group_ids = get_group_ids_for_replicates(y)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    class_names = le.classes_

    # Initialize cross-validation
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y_encoded, groups=group_ids)

    print(f"\nCross-validation: {n_splits} folds")
    print("Strategy: Train on 65 samples (5 replicates), test on 13 samples (1 replicate set)")

    # Store predictions
    y_true_all = []
    y_pred_all = []

    # Run cross-validation
    print("\nRunning cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_encoded, groups=group_ids)):
        if (fold + 1) % 10 == 0 or fold == 0:
            print(f"  Fold {fold + 1}/{n_splits}...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]  # Use string labels

        # Create and train model (model will encode labels internally)
        model = create_model()
        model.fit(X_train, y_train)

        # Predict (model returns encoded labels)
        y_pred_encoded = model.predict(X_test)

        # Encode true labels for comparison
        y_test_encoded = le.transform(y_test.astype(str))

        y_true_all.extend(y_test_encoded)
        y_pred_all.extend(y_pred_encoded)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Calculate metrics
    accuracy = np.mean(y_true_all == y_pred_all)

    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        mask = y_true_all == i
        if mask.sum() > 0:
            class_acc = np.mean(y_true_all[mask] == y_pred_all[mask])
            per_class_accuracy[class_name] = class_acc

    # Confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nPer-class Accuracy:")
    for class_name, acc in sorted(per_class_accuracy.items()):
        print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")

    # Prepare results for saving
    results = {
        "method": "leave_replicate_out",
        "description": "Leave-replicate-out cross-validation: each fold leaves out 1 complete replicate set (13 samples), trains on 65",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(class_names),
        "samples_per_class": 6,
        "n_folds": n_splits,
        "train_size_per_fold": 65,
        "test_size_per_fold": 13,
        "metrics": {
            "overall_accuracy": float(accuracy),
            "overall_accuracy_percent": float(accuracy * 100),
            "per_class_accuracy": {k: float(v) for k, v in per_class_accuracy.items()}
        },
        "confusion_matrix": cm.tolist()
    }

    # Save results
    save_results(results, output_dir)

    # Save confusion matrix plot
    # Resolve output path relative to project root
    if not Path(output_dir).is_absolute():
        full_output_dir = project_root / output_dir
    else:
        full_output_dir = Path(output_dir)

    cm_path = full_output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, str(cm_path))
    print(f"\nConfusion matrix saved to {cm_path}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if accuracy >= 0.90:
        print("✓ Excellent accuracy (>90%) indicates strong generalization")
        print("  The model learns oil type patterns, not individual samples")
    elif accuracy >= 0.80:
        print("✓ Good accuracy (80-90%) suggests reasonable generalization")
        print("  Model captures most oil type patterns")
    else:
        print("⚠ Lower accuracy suggests potential overfitting")
        print("  Model may be memorizing specific samples")

    return results


if __name__ == "__main__":
    run_leave_replicate_out()
