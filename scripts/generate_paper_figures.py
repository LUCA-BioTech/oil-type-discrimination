#!/usr/bin/env python3
"""
Generate all figures used in the paper.

This script creates publication-ready figures from the validation results
and additional analyses. Outputs are saved to paper/figures/.

Usage:
    python scripts/generate_paper_figures.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.utils import load_data
from src.visualization import (
    plot_nature_scatter_2d,
    plot_lda_scree,
    plot_feature_correlation_heatmap,
    plot_roc_by_lda_dimensions,
    plot_3d_with_confidence_ellipsoid,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


def main():
    """Generate all paper figures."""
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    # Load data
    X, y = load_data()
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")

    # Create output directory
    output_dir = Path("paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Nature-style 2D scatter plot (LDA-transformed)
    print("\n[1/6] Generating Nature-style 2D scatter plot...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda_2d = lda.fit_transform(X_scaled, y_encoded)

    model = GaussianNB()
    model.fit(X_lda_2d, y_encoded)
    y_pred = model.predict(X_lda_2d)
    y_pred_str = le.inverse_transform(y_pred)

    plot_nature_scatter_2d(
        X_lda_2d,
        y_pred_str,
        title="LDA 2D Visualization - Oil Type Classification",
        save_name=str(output_dir / "fig1_nature_scatter_2d")
    )
    print(f"   Saved: {output_dir / 'fig1_nature_scatter_2d.pdf'}")

    # Figure 2: LDA Scree plot
    print("\n[2/6] Generating LDA Scree plot...")
    _, _, _, scree_data = plot_lda_scree(
        X, y,
        output_path=str(output_dir / "fig2_lda_scree.pdf"),
        title="LDA Scree Plot - Variance Explained"
    )
    print(f"   Saved: {output_dir / 'fig2_lda_scree.pdf'}")

    # Export LDA Scree data to CSV
    import pandas as pd
    df_lda_scree = pd.DataFrame(scree_data)
    df_lda_scree.to_csv(output_dir / "fig2_lda_scree_data.csv", index=False)
    print(f"   Saved: {output_dir / 'fig2_lda_scree_data.csv'}")

    # Figure 3: Feature correlation heatmap
    print("\n[3/6] Generating feature correlation heatmap...")
    feature_names = [f"en{i}" for i in range(1, 16)]
    plot_feature_correlation_heatmap(
        X, feature_names,
        output_path=str(output_dir / "fig3_correlation_heatmap.pdf")
    )
    print(f"   Saved: {output_dir / 'fig3_correlation_heatmap.pdf'}")

    # Figure 4: ROC curves by LDA dimensions
    print("\n[4/6] Generating ROC curves...")
    _, results_by_dim = plot_roc_by_lda_dimensions(
        X, y, [2, 3, 4, 5],
        output_path=str(output_dir / "fig4_roc_curves.pdf")
    )
    print(f"   Saved: {output_dir / 'fig4_roc_curves.pdf'}")

    # Export ROC data to CSV
    import pandas as pd
    # AUC summary
    auc_summary = pd.DataFrame([
        {"lda_dim": n_dim, "auc": result["auc"]}
        for n_dim, result in results_by_dim.items()
    ])
    auc_summary.to_csv(output_dir / "fig4_roc_auc_summary.csv", index=False)
    print(f"   Saved: {output_dir / 'fig4_roc_auc_summary.csv'}")

    # ROC curves data
    roc_records = []
    for n_dim, result in results_by_dim.items():
        fpr = result["fpr"]
        tpr = result["tpr"]
        for i in range(len(fpr)):
            roc_records.append({
                "lda_dim": n_dim,
                "fpr": fpr[i],
                "tpr": tpr[i]
            })
    df_roc = pd.DataFrame(roc_records)
    df_roc.to_csv(output_dir / "fig4_roc_curves_data.csv", index=False)
    print(f"   Saved: {output_dir / 'fig4_roc_curves_data.csv'}")

    # Figure 5: 3D interactive visualization
    print("\n[5/6] Generating 3D interactive plot...")
    result = plot_3d_with_confidence_ellipsoid(
        X, y,
        output_path=str(output_dir / "fig5_3d_interactive.html")
    )
    if result is None:
        print("   Skipped: plotly not installed (install with: pip install plotly kaleido)")
    else:
        print(f"   Saved: {output_dir / 'fig5_3d_interactive.html'}")

    # Figure 6: Validation summary (from validation framework)
    print("\n[6/6] Validation summary already generated...")
    print("   See: experiments/validation_report/validation_summary.png")

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")

    # List actual files by type
    pdf_files = sorted(output_dir.glob("*.pdf"))
    svg_files = sorted(output_dir.glob("*.svg"))
    html_files = sorted(output_dir.glob("*.html"))
    csv_files = sorted(output_dir.glob("*.csv"))

    if pdf_files:
        print("\n  PDF figures:")
        for f in pdf_files:
            print(f"    - {f.name}")

    if svg_files:
        print("\n  SVG figures:")
        for f in svg_files:
            print(f"    - {f.name}")

    if csv_files:
        print("\n  Data files (CSV):")
        for f in csv_files:
            print(f"    - {f.name}")

    if html_files:
        print("\n  Interactive visualizations:")
        for f in html_files:
            print(f"    - {f.name}")

    print("\nValidation report:")
    print("  - experiments/validation_report/validation_summary.png")
    print("  - experiments/validation_report/validation_report.md")

    # Check if plotly is installed
    try:
        import plotly
    except ImportError:
        print("\nNote: Install plotly for 3D interactive visualization:")
        print("      pip install plotly kaleido")


if __name__ == "__main__":
    main()
