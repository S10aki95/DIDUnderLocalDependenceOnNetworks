"""
Xu Estimator: IPW Weights Computation

Common utility for computing IPW weights used in Xu (2025) IPW estimator.
"""

import numpy as np
from typing import Union


def compute_xu_ipw_weights(
    D: np.ndarray,
    G: np.ndarray,
    eta: np.ndarray,
    eta_1g: Union[float, np.ndarray],
    eta_0g: Union[float, np.ndarray],
    g_level: int,
    return_denominator: bool = False,
) -> np.ndarray:
    """Calculate IPW weights for Xu paper's DATT estimator

    Calculates weights for estimating DATT (Direct Average Treatment Effect on the Treated).
    Treatment group has weight 1, control group is weighted by odds ratio.

    Args:
        D: Treatment variable array
        G: Exposure level array
        eta: Array of propensity scores P(D=1|Z)
        eta_1g: Conditional probability P(G=g|D=1,Z) (scalar or array)
        eta_0g: Conditional probability P(G=g|D=0,Z) (scalar or array)
        g_level: Exposure level to calculate
        return_denominator: If True, return denominator array (kept for backward compatibility)

    Returns:
        IPW weight array (or denominator array)
    """
    # Indicator function I{G=g}
    I_Gg = (G == g_level).astype(float)

    # Avoid division by zero
    eta_safe = np.clip(eta, 1e-10, 1 - 1e-10)

    # Convert eta_1g and eta_0g to arrays (expand to all elements if scalar)
    if np.isscalar(eta_1g):
        eta_1g_array = np.full_like(D, eta_1g, dtype=float)
    else:
        eta_1g_array = np.asarray(eta_1g, dtype=float)

    if np.isscalar(eta_0g):
        eta_0g_array = np.full_like(D, eta_0g, dtype=float)
    else:
        eta_0g_array = np.asarray(eta_0g, dtype=float)

    eta_1g_safe = np.clip(eta_1g_array, 1e-10, 1 - 1e-10)
    eta_0g_safe = np.clip(eta_0g_array, 1e-10, 1 - 1e-10)

    # Calculate odds ratio (for DATT)
    odds = eta_safe / (1 - eta_safe)

    # Weights for DATT: treatment group is 1/eta_1g, control group is odds/eta_0g
    # weights = (D * I_Gg / eta_1g) - ((1-D) * odds * I_Gg / eta_0g)
    weights = (D * I_Gg / eta_1g_safe) - ((1 - D) * odds * I_Gg / eta_0g_safe)

    # For investigation: return denominator (kept for backward compatibility)
    if return_denominator:
        denominator = (
            eta_safe * (1 - eta_safe) * (D * eta_1g_safe + (1 - D) * eta_0g_safe)
        )
        return denominator

    return weights
