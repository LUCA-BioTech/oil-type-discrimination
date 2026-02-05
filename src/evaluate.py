"""
Evaluation script for oil type discrimination model.

Trains on the full dataset and generates confusion matrix visualization.
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import unified utilities
from src.utils import load_data, create_model


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Create and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix - Full Training Set", fontsize=14, pad=20)
    plt.colorbar(label='Count')

    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    """Train on full dataset and generate confusion matrix."""
    print("="*60)
    print("EVALUATION: Full Training Set Performance")
    print("="*60)

    # Load data using unified utility
    X, y = load_data("data/raw/data-923.csv")
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(set(y))} oil types")

    # Create and train model using unified utilities
    model = create_model()
    model.fit(X, y)

    # Predict on training set (returns encoded labels)
    y_pred_encoded = model.predict(X)

    # Decode predictions back to string labels for comparison
    class_names = model.get_classes()
    y_pred = class_names[y_pred_encoded]

    # Calculate accuracy
    accuracy = np.mean(y == y_pred)

    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y, y_pred))

    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred, labels=sorted(set(y)))
    class_names = sorted(set(y))

    # Create output directory if needed
    output_dir = Path("experiments/base_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix plot
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
