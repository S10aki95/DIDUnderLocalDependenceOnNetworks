"""
Experiment settings and parameter management

This module centrally manages parameters used in simulation experiments,
allowing users to easily change settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from scipy.stats import norm


@dataclass
class Config:
    """Unified configuration class"""

    # === Simulation Settings ===
    n_simulations: int = 100
    n_units: int = 500
    k_distance: float = 1.0
    space_size: float = 20.0
    random_seed: int = 42

    # === DGP Settings ===
    # Covariate generation
    z_std: float = 1.0
    z_u_correlation_base: float = 0.5

    # Treatment assignment
    treatment_z_coef: float = 0.3
    treatment_z_u_coef: float = 0.8

    # Outcome generation
    beta_1: float = 1.2
    beta_2: float = 0.5
    y1_error_std: float = 1.0
    delta: float = 1.0
    tau: float = 0.8  # Direct effect
    gamma_1: float = 0.1
    gamma_2: float = 0.2
    y2_error_std: float = 1.0
    spillover_effects: tuple = (0.0, 0.8, 1.6, 2.4)

    # === Estimator Settings ===
    ps_clip_min: float = 0.05
    ps_clip_max: float = 0.95
    logistic_max_iter: int = (
        500  # Maximum iterations for logistic regression (optimized to 500 for real data analysis speedup)
    )
    ols_min_samples: int = 24
    max_neighbor_features: int = 10
    # Neighbor sampling limit for AITT estimation (optimized to 10 for real data analysis speedup)
    max_neighbors_per_unit: int = 10
    epsilon: float = 1e-9
    confidence_level: float = 0.95
    verbose: bool = True  # Whether to output logs
    n_bootstrap: int = 100  # Number of bootstrap iterations (default: 100)
    use_bootstrap_se: bool = (
        False  # Whether to use bootstrap standard errors (default: False)
    )

    # === HAC Standard Error Settings ===
    hac_bandwidth_multiplier: float = 2.0  # Bandwidth = hac_bandwidth_multiplier * K
    hac_default_max_distance: float = (
        1.0  # Default maximum distance when using distance calculation function
    )

    # === Distance Settings for Real Data Analysis ===
    county_distance_same: float = 0.3  # Distance between units in the same county
    county_distance_different: float = (
        1.0  # Distance between units in different counties
    )

    # === Report Generation Settings ===
    # Estimators to compare in coverage rate table (based on README.md requirements)
    coverage_table_estimators: list = None

    def __post_init__(self):
        """Post-initialization processing"""
        if self.coverage_table_estimators is None:
            self.coverage_table_estimators = [
                "Proposed IPW (ADTT)",
                "Proposed IPW (AITT)",
                "Proposed DR (ADTT)",
                "Proposed DR (AITT)",
                "Canonical IPW",
                "Canonical TWFE",
                "DR-DID",
                "Modified TWFE",
                "Xu DR (CS) - ODE",
                "Xu DR (MO) - ODE",
                "Xu DR (FM) - ODE",
                "Xu IPW (CS) - ODE",
                "Xu IPW (MO) - ODE",
                "Xu IPW (FM) - ODE",
            ]

        # Dynamically calculate z_critical based on confidence_level
        self.z_critical = norm.ppf(1 - (1 - self.confidence_level) / 2)

    # === Visualization settings ===
    figsize: tuple = (12, 8)
    dpi: int = 300
    title_fontsize: int = 20
    label_fontsize: int = 12
    legend_fontsize: int = 10


# Robustness scenario definitions (simplified as dictionary)
ROBUSTNESS_SCENARIOS = {
    "sample_size": {
        "parameter": "n_units",
        "values": [300, 500, 700],
        "description": "Robustness to sample size variation",
        "display_name": "Sample Size",
    },
    "network_density": {
        "parameter": "k_distance",
        "values": [0.8, 1.0, 1.2],
        "description": "Robustness to network density (interference range) variation",
        "display_name": "Network Density",
    },
    "spatial_correlation": {
        "parameter": "z_u_correlation_base",
        "values": [0.2, 0.5, 0.8],
        "description": "Robustness to spatial correlation strength variation",
        "display_name": "Spatial Correlation",
    },
    "neighbor_features": {
        "parameter": "max_neighbor_features",
        "values": [5, 10, 15],
        "description": "Robustness to neighbor feature count variation",
        "display_name": "Neighbor Features",
    },
}


def get_config(
    config_name: str = "default", overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Return configuration based on configuration name (with override functionality)

    Args:
        config_name: Base configuration name ("default", "robustness", "simulation")
            - "default": Optimized settings for real data analysis (speedup)
            - "simulation": High-precision settings for simulation experiments
            - "robustness": For robustness checks (same as "default")
        overrides: Dictionary of settings to override

    Returns:
        Configuration object
    """
    # Base configuration
    if config_name == "simulation":
        # For simulation experiments: higher precision settings (original values)
        config = Config(
            max_neighbors_per_unit=20,  # Use more neighbors in simulation
            logistic_max_iter=1000,  # Use more iterations in simulation
        )
    else:
        # Default settings (optimized for real data analysis)
        # Note: If high precision is needed for simulation experiments,
        # explicitly override with overrides or use config_name="simulation"
        config = Config()

    # Apply override processing
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' - skipping")

    return config


def get_robustness_scenarios() -> Dict[str, Dict[str, Any]]:
    """Get robustness check scenarios"""
    return ROBUSTNESS_SCENARIOS
