# Paper Figures and Materials

This directory contains all figures and materials used in the paper.

## Figures

| Figure | File | Description | Generation |
|--------|------|-------------|------------|
| Fig 1 | `fig1_nature_scatter_2d.pdf` | LDA-transformed 2D scatter plot | `scripts/generate_paper_figures.py` or `notebooks/iol_classification.ipynb` |
| Fig 2 | `fig2_lda_scree.pdf` | LDA Scree plot | `scripts/generate_paper_figures.py` or `notebooks/iol_classification.ipynb` |
| Fig 3 | `fig3_correlation_heatmap.pdf` | Feature correlation matrix | `scripts/generate_paper_figures.py` or `notebooks/iol_classification.ipynb` |
| Fig 4 | `fig4_roc_curves.pdf` | ROC curves by LDA dimensions | `scripts/generate_paper_figures.py` or `notebooks/iol_classification.ipynb` |
| Fig 5 | `fig5_3d_interactive.html` | 3D interactive visualization | `scripts/generate_paper_figures.py` or `notebooks/iol_classification.ipynb` |
| Fig 6 | `fig6_validation_summary.png` | Comprehensive validation summary | `src/validation/run_all_validations.py` |

## Regenerating Figures

### To regenerate all figures:
```bash
python scripts/generate_paper_figures.py
```

### To regenerate specific figures, run the corresponding notebooks:
```bash
jupyter notebook notebooks/iol_classification.ipynb
jupyter notebook notebooks/model.ipynb
```

## Figure Quality

- **Vector graphics (PDF)**: Suitable for publication
- **Interactive HTML**: For online supplementary materials
- **Resolution**: 300 DPI or higher

## Data Sources

All figures use the same dataset:
- File: `../data/raw/data-923.csv`
- Samples: 78 (13 oil types × 6 replicates)
- Features: 15 enzyme absorbance values

## Validation Results

Comprehensive validation results are in `../experiments/validation_report/`:
- `validation_summary.png` - Combined validation summary
- `validation_report.md` - Detailed validation report

## Notebooks Documentation

### `iol_classification.ipynb`
- **Purpose**: Exploratory data analysis, algorithm comparison, and publication-ready visualizations
- **Algorithms Tested**: Decision Tree, KNN, GaussianNB, MLP, LDA, CatBoost, Extra Trees
- **Key Outputs**: Nature-style plots, ROC curves, 3D visualizations

### `model.ipynb`
- **Purpose**: Model validation and generalization analysis
- **Analyses**: 5-Fold CV, Permutation Test, LDA Scree Plot
- **Key Outputs**: Validation metrics, statistical significance tests
