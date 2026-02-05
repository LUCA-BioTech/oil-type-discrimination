"""
Interactive 3D visualizations using Plotly.

This module contains functions for creating interactive 3D scatter plots
with confidence ellipsoids for oil type discrimination.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    px = None


def get_confidence_ellipsoid(x, y, z, confidence=0.9):
    """
    Generate points for a 3D confidence ellipsoid.

    Args:
        x, y, z: Coordinate arrays
        confidence: Confidence level (default 0.9 for 90%)

    Returns:
        tuple: (ex, ey, ez) arrays of ellipsoid surface points, or (None, None, None) if insufficient data
    """
    data = np.vstack((x, y, z))
    if data.shape[1] < 4:
        return None, None, None

    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    eigvals, eigvecs = np.linalg.eigh(cov)
    scale_factor = np.sqrt(chi2.ppf(confidence, df=3))

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    sphere_points = np.stack((
        x_sphere.flatten(),
        y_sphere.flatten(),
        z_sphere.flatten()
    ))

    radii = np.sqrt(np.maximum(eigvals, 0)) * scale_factor
    transformed = (eigvecs @ np.diag(radii) @ sphere_points).T + mean

    return transformed[:, 0], transformed[:, 1], transformed[:, 2]


def plot_3d_with_confidence_ellipsoid(
    X,
    y,
    output_path="3D_interactive.html",
    test_size=0.3,
    confidence=0.9,
    random_state=42
):
    """
    Interactive 3D scatter plot with 90% confidence ellipsoids.

    This function creates an interactive 3D visualization using Plotly,
    showing LDA-transformed data with confidence ellipsoids for each class.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        output_path: Path to save the HTML file
        test_size: Fraction of data to use as test set
        confidence: Confidence level for ellipsoids (default 0.9)
        random_state: Random seed for reproducibility

    Returns:
        plotly.graph_objects.Figure: The interactive figure, or None if plotly is not installed

    From: notebooks/iol_classification.ipynb
    """
    if not HAS_PLOTLY:
        return None

    # Data preparation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    df_data = pd.DataFrame(X_scaled)
    df_data['cate_encoded'] = le.fit_transform(y.astype(str))

    y_encoded = df_data['cate_encoded'].values
    X_scaled_array = df_data[[f"en{i}" for i in range(1, 16)]].values if hasattr(X, 'columns') else X_scaled

    # LDA dimensionality reduction
    lda = LinearDiscriminantAnalysis(n_components=min(3, len(np.unique(y_encoded)) - 1))
    X_reduced_all = lda.fit_transform(X_scaled_array, y_encoded)
    X_plot_3d = X_reduced_all[:, :3]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced_all, y_encoded,
        test_size=test_size, stratify=y_encoded,
        random_state=random_state
    )

    # Train model
    model = GaussianNB(var_smoothing=1e-2)
    model.fit(X_train, y_train)

    # Calculate test set metrics
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)

    metric_map = {}
    unique_labels_encoded = np.unique(y_test)
    unique_labels_str = le.inverse_transform(unique_labels_encoded)

    for i, label_str in enumerate(unique_labels_str):
        tp = cm[i, i]
        total_true = cm[i, :].sum()
        recall = tp / total_true if total_true > 0 else 0
        total_pred = cm[:, i].sum()
        precision = tp / total_pred if total_pred > 0 else 0
        metric_map[label_str] = f" | R:{recall:.2f}, P:{precision:.2f}"

    # Prepare plotting data
    all_probs = model.predict_proba(X_reduced_all)
    confidence_scores = np.max(all_probs, axis=1)
    y_pred = model.predict(X_reduced_all)
    y_pred_str = le.inverse_transform(y_pred)
    y_true_str = le.inverse_transform(y_encoded)

    df_plot = pd.DataFrame({
        "Comp1": X_plot_3d[:, 0],
        "Comp2": X_plot_3d[:, 1],
        "Comp3": X_plot_3d[:, 2],
        "True Label": y_true_str,
        "Predicted Label": y_pred_str,
        "Confidence": confidence_scores,
        "Is_Correct": y_encoded == y_pred
    })
    df_plot["Status"] = np.where(df_plot["Is_Correct"], "Correct", "Misclassified")

    # Create figure
    fig = go.Figure()

    # Add misclassified legend marker
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(symbol='diamond-open', color='rgba(0,0,0,0.5)', size=8),
        name='Misclassified Points',
        showlegend=True,
        legendgroup='_misc',
        hoverinfo='skip'
    ))

    # Generate colors
    num_classes = len(unique_labels_str)
    colors_base = px.colors.qualitative.Alphabet if num_classes > 10 else px.colors.qualitative.Plotly
    colors_extended = colors_base * ((num_classes // len(colors_base)) + 1)
    color_map = {label: colors_extended[i] for i, label in enumerate(unique_labels_str)}

    # Add traces for each class
    for label in unique_labels_str:
        subset = df_plot[df_plot["True Label"] == label]
        color = color_map[label]
        metrics_suffix = metric_map.get(label, "")

        # 90% confidence ellipsoid
        ex, ey, ez = get_confidence_ellipsoid(
            subset["Comp1"].values,
            subset["Comp2"].values,
            subset["Comp3"].values,
            confidence=confidence
        )

        if ex is not None:
            # Add ellipsoid mesh
            fig.add_trace(go.Mesh3d(
                x=ex, y=ey, z=ez,
                alphahull=0, opacity=0.15, color=color,
                name=f"{label}{metrics_suffix}",
                legendgroup=label,
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add invisible scatter for legend
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(color=color, symbol='circle', size=10),
                name=f"{label}{metrics_suffix}",
                legendgroup=label,
                showlegend=True,
                hoverinfo='skip',
                visible='legendonly',
                legendrank=0
            ))

        # Correctly classified points
        sub_subset_correct = subset[subset["Status"] == "Correct"]
        if not sub_subset_correct.empty:
            custom_data_correct = np.stack((
                sub_subset_correct["True Label"],
                sub_subset_correct["Predicted Label"],
                sub_subset_correct["Status"],
                sub_subset_correct["Confidence"]
            ), axis=-1)

            fig.add_trace(go.Scatter3d(
                x=sub_subset_correct["Comp1"],
                y=sub_subset_correct["Comp2"],
                z=sub_subset_correct["Comp3"],
                mode='markers',
                name=f"{label} (Correct)",
                legendgroup=label,
                showlegend=False,
                marker=dict(
                    size=sub_subset_correct["Confidence"] * 10 + 2,
                    color=color,
                    symbol='circle',
                    opacity=0.9,
                    line=dict(width=0)
                ),
                customdata=custom_data_correct,
                hovertemplate=(
                    "<b>True Label: %{customdata[0]}</b><br>" +
                    "Predicted: %{customdata[1]}<br>" +
                    "Status: %{customdata[2]}<br>" +
                    "Confidence: %{customdata[3]:.1%}<br>" +
                    "<extra></extra>"
                )
            ))

        # Misclassified points
        sub_subset_misclassified = subset[subset["Status"] == "Misclassified"]
        if not sub_subset_misclassified.empty:
            custom_data_misclassified = np.stack((
                sub_subset_misclassified["True Label"],
                sub_subset_misclassified["Predicted Label"],
                sub_subset_misclassified["Status"],
                sub_subset_misclassified["Confidence"]
            ), axis=-1)

            fig.add_trace(go.Scatter3d(
                x=sub_subset_misclassified["Comp1"],
                y=sub_subset_misclassified["Comp2"],
                z=sub_subset_misclassified["Comp3"],
                mode='markers',
                name=f"{label} (Misclassified)",
                legendgroup='_misc',
                showlegend=False,
                marker=dict(
                    size=sub_subset_misclassified["Confidence"] * 10 + 2,
                    color=color,
                    symbol='diamond-open',
                    opacity=0.9,
                    line=dict(width=0)
                ),
                customdata=custom_data_misclassified,
                hovertemplate=(
                    "<b>True Label: %{customdata[0]}</b><br>" +
                    "Predicted: %{customdata[1]}<br>" +
                    "Status: %{customdata[2]}<br>" +
                    "Confidence: %{customdata[3]:.1%}<br>" +
                    "<extra></extra>"
                )
            ))

    # Layout
    fig.update_layout(
        title="3D Analysis: Per-Class Test Set Metrics in Legend",
        scene=dict(
            xaxis_title='LDA Component 1',
            yaxis_title='LDA Component 2',
            zaxis_title='LDA Component 3',
            aspectmode='cube'
        ),
        width=1200,
        height=900,
        legend=dict(
            title="Class (R=Recall, P=Precision)",
            itemsizing='constant',
            groupclick="toggleitem"
        )
    )

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to HTML
    fig.write_html(output_path)
    print(f"   Saved: {output_path}")

    return fig
