"""
Visualization and report generation module

Performs visualization of simulation results and Markdown report generation.
"""

# Configuration and constants
from .config import VisualizationConfig, get_labels

# Common utilities
from .utils import extract_estimator_data, setup_plot_style, save_plot

# Markdown generation
from .markdown import generate_results_markdown

# Basic visualization functions
from .plots import (
    create_spillover_plot,
    create_units_scatter_plot,
    create_distribution_plot_adtt,
    create_distribution_plot_aitt,
    create_coverage_rate_table,
    create_influence_function_comparison_plot,
    create_se_comparison_table,
    create_bootstrap_se_distribution_plot,
)

# Robustness visualization functions
from .robustness import (
    create_robustness_line_plot,
    create_sensitivity_line_plot,
    create_robustness_plots,
)

# Helper functions
from .helpers import (
    classify_experiments_by_type,
    map_estimator_names,
    cleanup_old_results,
)

# Weight analysis functions
from .weight_analysis import (
    collect_influence_functions,
    collect_influence_functions_from_simulation,
)

__all__ = [
    # Configuration and constants
    "VisualizationConfig",
    "get_labels",
    # Common utilities
    "extract_estimator_data",
    "setup_plot_style",
    "save_plot",
    # Markdown generation
    "generate_results_markdown",
    # Basic visualization functions
    "create_spillover_plot",
    "create_units_scatter_plot",
    "create_distribution_plot_adtt",
    "create_distribution_plot_aitt",
    "create_coverage_rate_table",
    "create_influence_function_comparison_plot",
    "create_se_comparison_table",
    "create_bootstrap_se_distribution_plot",
    # Robustness visualization functions
    "create_robustness_line_plot",
    "create_sensitivity_line_plot",
    "create_robustness_plots",
    # Helper functions
    "classify_experiments_by_type",
    "map_estimator_names",
    "cleanup_old_results",
    # Weight analysis functions
    "collect_influence_functions",
    "collect_influence_functions_from_simulation",
]
