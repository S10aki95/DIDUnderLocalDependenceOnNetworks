"""
Xu Estimator: DR Weights Computation

Common utility for computing DR weights used in Xu (2025) DR estimator.
"""

import numpy as np
from typing import Tuple


def compute_xu_dr_weights(
    D: np.ndarray,
    G: np.ndarray,
    eta: np.ndarray,
    eta_1g: np.ndarray,
    eta_0g: np.ndarray,
    m_1g_pred: np.ndarray,
    m_0g_pred: np.ndarray,
    g_level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate weights for Xu paper's DR estimator (for DATT)

    Calculates weights for estimating DATT (Direct Average Treatment Effect on the Treated).
    Treatment group has weight 1, control group is weighted by odds ratio.

    Args:
        D: Treatment variable array
        G: Exposure level array
        eta: Array of propensity scores P(D=1|Z)
        eta_1g: Array of conditional probabilities P(G=g|D=1,Z)
        eta_0g: Array of conditional probabilities P(G=g|D=0,Z)
        m_1g_pred: Array of outcome predictions for treatment group
        m_0g_pred: Array of outcome predictions for control group
        g_level: Exposure level to calculate

    Returns:
        Tuple of (weights_for_delta_Y, adjustment)
    """
    # Indicator function I{G=g}
    I_Gg = (G == g_level).astype(float)

    # Avoid division by zero
    eta_safe = np.clip(eta, 1e-10, 1 - 1e-10)
    eta_1g_safe = np.clip(eta_1g, 1e-10, 1 - 1e-10)
    eta_0g_safe = np.clip(eta_0g, 1e-10, 1 - 1e-10)

    # Calculate odds ratio (for DATT)
    odds = eta_safe / (1 - eta_safe)

    # Weights for DATT: treatment group has weight 1, control group weighted by odds ratio
    # weights_for_delta_Y = (D * I_Gg / eta_1g) - ((1-D) * odds * I_Gg / eta_0g)
    weights_for_delta_Y = (D * I_Gg / eta_1g_safe) - (
        (1 - D) * odds * I_Gg / eta_0g_safe
    )

    # Adjustment term (for DATT): - (D * I_Gg / eta_1g) * m_1g_pred + ((1-D) * odds * I_Gg / eta_0g) * m_0g_pred + D * (m_1g_pred - m_0g_pred)
    # Note: m_1g_pred - m_0g_pred is applied only to treatment group (multiply by D)
    adjustment = (
        -(D * I_Gg / eta_1g_safe) * m_1g_pred
        + ((1 - D) * odds * I_Gg / eta_0g_safe) * m_0g_pred
        + D * (m_1g_pred - m_0g_pred)
    )

    return weights_for_delta_Y, adjustment
