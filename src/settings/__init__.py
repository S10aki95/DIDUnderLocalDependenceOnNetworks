"""
Settings module for configuration and data generation

This module provides configuration management and data generation utilities
for both simulation and real data analysis.
"""

# Config related
from .config import (
    Config,
    get_config,
    get_robustness_scenarios,
    ROBUSTNESS_SCENARIOS,
)

# Common functions
from .functions import (
    print_config_summary,
    print_robustness_summary,
)

# DGP for simulation
from .simulation import (
    generate_data,
    calculate_conditional_params,
)

# Loader for real data
from .real_data_sez import (
    SEZDataLoader,
    load_sez_data,
)

__all__ = [
    # Config
    "Config",
    "get_config",
    "get_robustness_scenarios",
    "ROBUSTNESS_SCENARIOS",
    # Functions
    "print_config_summary",
    "print_robustness_summary",
    # Simulation
    "generate_data",
    "calculate_conditional_params",
    # Real data
    "SEZDataLoader",
    "load_sez_data",
]
