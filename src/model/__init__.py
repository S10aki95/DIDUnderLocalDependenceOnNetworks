"""
Model estimators package

This package contains different estimation methods organized by approach:
- standard: Standard estimation methods (no interference)
- xu: Xu (2025) estimators
- proposed: Proposed nonparametric interference methods
- common: Common utilities (HAC standard errors, etc.)
"""

# Standard estimation methods
from .standard import (
    estimate_twfe,
    estimate_ipw,
    estimate_modified_twfe,
    estimate_dr_did,
)

# Xu (2025) estimators
from .xu import (
    estimate_xu_propensity_scores,
    compute_xu_ipw_weights,
    compute_xu_ipw_influence_function,
    compute_xu_dr_weights,
    compute_xu_dr_influence_function,
    estimate_xu_dr,
    estimate_xu_ipw,
)

# Proposed methods
from .proposed import (
    compute_adtt_influence_function,
    compute_aitt_influence_function,
    estimate_proposed_with_se,
)

# Common utilities
from .common import estimate_hac_se

__all__ = [
    # Standard
    "estimate_twfe",
    "estimate_ipw",
    "estimate_modified_twfe",
    "estimate_dr_did",
    # Xu
    "estimate_xu_propensity_scores",
    "compute_xu_ipw_weights",
    "compute_xu_ipw_influence_function",
    "compute_xu_dr_weights",
    "compute_xu_dr_influence_function",
    "estimate_xu_dr",
    "estimate_xu_ipw",
    # Proposed
    "compute_adtt_influence_function",
    "compute_aitt_influence_function",
    "estimate_proposed_with_se",
    # Common
    "estimate_hac_se",
]
