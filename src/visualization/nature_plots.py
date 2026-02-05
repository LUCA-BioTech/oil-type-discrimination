"""
Nature-style publication plots for oil type discrimination.

This module contains functions to create publication-ready 2D scatter plots
and confusion matrices following Nature journal style guidelines.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_nature_scatter(
    X2d,
    y_labels,
    title="Nature-style Plot",
    save_name=None,
    figsize=(6.5, 5)
):
    """
    Create Nature-style 2D scatter plot.

    This function creates a publication-quality scatter plot following
    Nature journal style guidelines with clean axes and legends.

    Args:
        X2d: 2D array of shape (n_samples, 2) - reduced dimensional data
        y_labels: Array of shape (n_samples,) - class labels
        title: Plot title
        save_name: Path to save the figure (without extension). Saves as PDF and SVG.
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object

    From: notebooks/iol_classification.ipynb
    """
    plt.rcParams.update({
        "font.family": "Arial",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": False,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
    })

    nature_colors = [
        "#4C72B0", "#DD8452", "#55A868",
        "#C44E52", "#8172B3", "#937860",
        "#DA8BC3", "#8C8C8C", "#64B5CD"
    ]

    fig = plt.figure(figsize=figsize)
    unique_classes = np.unique(y_labels)

    for i, cls in enumerate(unique_classes):
        idx = (y_labels == cls)
        plt.scatter(
            X2d[idx, 0], X2d[idx, 1],
            s=55, alpha=0.78,
            color=nature_colors[i % len(nature_colors)],
            edgecolor="black", linewidth=0.35,
            label=str(cls)
        )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Component 1", labelpad=6)
    plt.ylabel("Component 2", labelpad=6)
    plt.title(title, pad=10)

    # Legend on the right side
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.3
    )

    plt.tight_layout()

    if save_name:
        plt.savefig(f"{save_name}.pdf", bbox_inches="tight")
        plt.savefig(f"{save_name}.svg", bbox_inches="tight")

    return fig


def plot_confusion_matrix_advanced(
    cm,
    classes,
    output_path=None,
    figsize=(8, 6),
    diag_color="#ff7f0e"
):
    """
    Create advanced confusion matrix with diagonal highlighting.

    This function creates a publication-quality confusion matrix with
    highlighted diagonal elements for correct predictions.

    Args:
        cm: Confusion matrix array from sklearn.metrics.confusion_matrix
        classes: List of class names
        output_path: Path to save the figure (with extension)
        figsize: Figure size (width, height) in inches
        diag_color: Color for diagonal elements (correct predictions)

    Returns:
        matplotlib Figure object

    From: notebooks/iol_classification.ipynb
    """
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # Draw diagonal rectangles
    for i in range(n_classes):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=True, color=diag_color, alpha=0.8
        ))

    # Add value annotations
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j + 0.5, i + 0.5, str(cm[i, j]),
                ha='center', va='center',
                fontsize=10, color='black'
            )

    # Set ticks and labels
    ax.set_xticks(np.arange(n_classes) + 0.5)
    ax.set_yticks(np.arange(n_classes) + 0.5)
    ax.set_xticklabels(classes, ha='right', fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    # Axis labels and title
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13)

    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(0, n_classes)
    ax.set_ylim(0, n_classes)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    ax.grid(
        linestyle='--',
        alpha=0.7,
        color='lightgray',
        linewidth=0.3
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
