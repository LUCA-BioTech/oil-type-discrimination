"""
Statistical analysis plots for oil type discrimination.

This module contains functions for LDA analysis, feature correlation,
and ROC curve visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    # Fallback colors - mimicking plotly structure
    class _FallbackQualitative:
        Plotly = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a']

    class _FallbackColors:
        qualitative = _FallbackQualitative()

    class _FallbackPx:
        colors = _FallbackColors()

    px = _FallbackPx()


def plot_lda_scree(
    X,
    y,
    output_path=None,
    figsize=(6, 4),
    title="LDA Scree Plot"
):
    """
    LDA Scree plot with dual y-axis (bar + cumulative line).

    This function creates a publication-quality LDA scree plot showing
    the variance explained by each LDA component and the cumulative
    explained variance.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        output_path: Path to save the figure (with extension)
        figsize: Figure size (width, height) in inches
        title: Plot title

    Returns:
        tuple: (fig, ax1, ax2, scree_data) matplotlib Figure, Axes objects, and scree data dict

    From: notebooks/iol_classification.ipynb, notebooks/model.ipynb
    """
    # Fit LDA to get explained variance ratio
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, y_encoded)

    explained_var = lda.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    ld_indices = np.arange(1, len(explained_var) + 1)

    # Prepare scree data for export
    scree_data = {
        'LD': ld_indices,
        'explained_variance_ratio': explained_var,
        'cumulative_explained_variance': cum_var,
        'cumulative_explained_variance_%': cum_var * 100
    }

    fig, ax1 = plt.subplots(figsize=figsize)

    # Left axis: bar plot
    bars = ax1.bar(
        ld_indices, explained_var,
        color="#1f77b4", alpha=0.85,
        edgecolor='k', linewidth=0,
        label="Individual LD"
    )
    ax1.set_xlabel("LDA Component", fontsize=10)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=10, color="#1f77b4")
    ax1.tick_params(axis='y', labelcolor="#1f77b4", labelsize=9, width=0.5, length=4)
    ax1.tick_params(axis='x', width=0.5, length=4)
    ax1.set_xticks(ld_indices)
    ax1.set_ylim(0, max(explained_var) * 1.3)

    # Annotate each bar with contribution
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2., 1.02 * height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=7
        )

    # Right axis: cumulative line
    ax2 = ax1.twinx()
    ax2.plot(
        ld_indices, cum_var * 100,
        color="#ff7f0e", marker='o', markersize=5,
        linewidth=1.8, label="Cumulative Explained"
    )
    ax2.set_ylabel(
        "Cumulative Explained Variance (%)",
        fontsize=10, color="#ff7f0e"
    )
    ax2.tick_params(axis='y', labelcolor="#ff7f0e", labelsize=9, width=0.5, length=4)
    ax2.set_ylim(0, 105)

    # Mark 80% variance threshold
    threshold = 80
    first_above = np.where(cum_var * 100 >= threshold)[0]
    if len(first_above) > 0:
        first_above = first_above[0] + 1
        ax2.axvline(x=first_above, color='grey', linestyle='--', linewidth=0.5)
        ax2.text(
            first_above + 0.3, 30,
            f'{threshold}% variance at LD{first_above}',
            color='grey', fontsize=8
        )

    # Thin axis lines
    for spine in ax1.spines.values():
        spine.set_linewidth(0.2)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.2)

    ax1.grid(linestyle='-', alpha=0.7, color='lightgray', linewidth=0.3)
    ax2.grid(linestyle='-', alpha=0.7, color='lightgray', linewidth=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax1, ax2, scree_data


def plot_feature_correlation_heatmap(
    X,
    feature_names,
    output_path=None,
    figsize=(10, 8)
):
    """
    Feature correlation matrix with upper triangle colors + lower triangle values.

    This function creates a publication-quality correlation heatmap with
    colors in the upper triangle and correlation values in the lower triangle.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        feature_names: List of feature names
        output_path: Path to save the figure (with extension)
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object

    From: notebooks/iol_classification.ipynb
    """
    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)

    # Upper triangle heatmap (colors)
    sns.heatmap(
        corr_matrix,
        mask=~mask,
        annot=False,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        cbar=True,
        linewidths=0.3,
        linecolor='gray',
        square=True,
        ax=ax
    )

    # Lower triangle values
    for i in range(corr_matrix.shape[0]):
        for j in range(i):
            ax.text(
                j + 0.5, i + 0.5,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha='center', va='center',
                fontsize=9, color='black'
            )

    # Axis settings
    ax.set_xticks(np.arange(corr_matrix.shape[0]) + 0.5)
    ax.set_yticks(np.arange(corr_matrix.shape[0]) + 0.5)
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr_matrix.columns, rotation=0, fontsize=10)
    ax.set_title("Feature Correlation Matrix", fontsize=13)

    # Thin borders
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    ax.grid(linestyle='-', alpha=0.7, color='lightgray', linewidth=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_by_lda_dimensions(
    X,
    y,
    lda_dimensions=None,
    output_path=None,
    figsize=(8, 6),
    n_folds=5,
    random_state=42
):
    """
    ROC curves comparison across different LDA dimensions.

    This function creates a publication-quality plot showing ROC curves
    for different LDA dimensionalities using cross-validation.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        lda_dimensions: List of LDA dimensions to test (default: [2,3,4,5])
        output_path: Path to save the figure (with extension)
        figsize: Figure size (width, height) in inches
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        tuple: (fig, results_by_dim) matplotlib Figure and results dictionary.
               results_by_dim contains fpr, tpr, auc, and color for each LDA dimension.

    From: notebooks/iol_classification.ipynb
    """
    if lda_dimensions is None:
        lda_dimensions = [2, 3, 4, 5]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    class_names = le.classes_
    n_classes = len(class_names)
    max_lda_dim = n_classes - 1

    # Filter valid dimensions
    lda_dimensions = [d for d in lda_dimensions if d <= max_lda_dim]

    results_by_dim = {}
    colors_extended = px.colors.qualitative.Plotly * 2

    # Test each LDA dimension
    for n_dim in lda_dimensions:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        y_true_all_cv, y_prob_all_cv = [], []

        for train_idx, test_idx in kf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # LDA dimensionality reduction
            lda = LinearDiscriminantAnalysis(n_components=n_dim)
            X_train_lda = lda.fit_transform(X_train, y_train)
            X_test_lda = lda.transform(X_test)

            # GaussianNB model
            model = GaussianNB()
            model.fit(X_train_lda, y_train)

            y_prob = model.predict_proba(X_test_lda)
            y_true_all_cv.append(y_test)
            y_prob_all_cv.append(y_prob)

        # Aggregate CV results
        y_true_all = np.concatenate(y_true_all_cv)
        y_prob_all = np.concatenate(y_prob_all_cv)

        # Calculate overall AUC and ROC curve
        y_true_bin = label_binarize(y_true_all, classes=np.arange(n_classes))

        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob_all.ravel())
        roc_auc = auc(fpr, tpr)

        results_by_dim[n_dim] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'color': colors_extended[n_dim - 1]
        }

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=0.8)

    best_auc = -1
    best_dim = 0

    for n_dim, result in results_by_dim.items():
        label = f"LDA Dim: {n_dim} (AUC={result['auc']:.3f})"
        ax.plot(
            result['fpr'], result['tpr'],
            color=result['color'],
            lw=1.5,
            label=label,
            alpha=0.7
        )

        if result['auc'] > best_auc:
            best_auc = result['auc']
            best_dim = n_dim

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curve Comparison ({n_folds}-Fold CV) - Best Dim: {best_dim} (AUC={best_auc:.3f})",
        fontsize=14
    )
    ax.tick_params(axis='both', labelsize=11, width=0.5, length=4)

    # Thin axis lines
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    ax.legend(
        fontsize=9, loc="lower right",
        frameon=True, framealpha=0.95, ncols=2
    )

    ax.grid(linestyle='--', alpha=0.5, color='lightgray', linewidth=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, results_by_dim
