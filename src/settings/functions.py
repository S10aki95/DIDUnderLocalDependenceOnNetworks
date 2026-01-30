"""
Common functions module

Provides common processing such as setting display.
"""

from typing import Optional
from .config import Config, get_config, ROBUSTNESS_SCENARIOS


def print_config_summary(config: Optional[Config] = None) -> None:
    """Display configuration summary"""
    if config is None:
        config = get_config("default")

    print("=== Configuration Summary ===")
    print(f"Simulations: {config.n_simulations}")
    print(f"Units: {config.n_units}")
    print(f"Neighbor Distance (K): {config.k_distance}")
    print(f"Space Size: {config.space_size}")
    print(f"Random Seed: {config.random_seed}")
    print(f"Direct Effect (τ): {config.tau}")
    print(f"Spillover Effects: {config.spillover_effects}")
    print("=============================")


def print_robustness_summary() -> None:
    """Display robustness settings summary"""
    print("=== Robustness Scenarios ===")
    for scenario_name, scenario in ROBUSTNESS_SCENARIOS.items():
        print(f"{scenario_name}:")
        print(f"  Parameter: {scenario['parameter']}")
        print(f"  Values: {scenario['values']}")
        print(f"  Description: {scenario['description']}")
        print()
    print("=============================")
