"""
Visualization module for oil type discrimination project.

Contains publication-ready plotting functions extracted from notebooks.
"""

from .nature_plots import plot_nature_scatter, plot_confusion_matrix_advanced
from .statistical_plots import (
    plot_lda_scree,
    plot_feature_correlation_heatmap,
    plot_roc_by_lda_dimensions,
)
from .interactive_plots import plot_3d_with_confidence_ellipsoid

__all__ = [
    'plot_nature_scatter',
    'plot_confusion_matrix_advanced',
    'plot_lda_scree',
    'plot_feature_correlation_heatmap',
    'plot_roc_by_lda_dimensions',
    'plot_3d_with_confidence_ellipsoid',
]

# Alias for backward compatibility
plot_nature_scatter_2d = plot_nature_scatter
__all__.append('plot_nature_scatter_2d')
