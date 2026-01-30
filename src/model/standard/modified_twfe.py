"""
Modified TWFE (Modified Two-Way Fixed Effects) Estimator

Standard estimation method with exposure mapping based on Xu (2025).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from typing import List
from ...utils import prepare_data_for_estimation
from ..common.models import ATTEstimateResult


def estimate_modified_twfe(
    df: pd.DataFrame, neighbors_list: List[list]
) -> ATTEstimateResult:
    """Calculate Modified TWFE estimator

    Based on definition in Xu (2025). Model formula:
    delta_Y = β_0 + τ_1 * D + τ_2 * (1-D) * S_i + τ_3 * D * S_i + β_1' * Z + ε

    where S_i is a binary variable indicating presence (1) or absence (0) of treated units in neighbors.

    Args:
        df: Dataframe
        neighbors_list: Neighbor list

    Returns:
        ODE estimate and standard error (NaN on failure)
    """
    df_est, _ = prepare_data_for_estimation(df)

    # Calculate number of treated units in neighbors
    s_i_counts = np.array([df["D"].iloc[n].sum() for n in neighbors_list])
    # Convert to binary variable indicating presence/absence of exposure based on paper definition
    s_i_binary = (s_i_counts > 0).astype(int)
    df_est["S_i"] = s_i_binary

    # Modified TWFE model: ΔY_i = β_0 + τ_1 D_i + τ_2 (1-D_i)S_i + τ_3 D_i S_i + β_1' Z_i + ε_i
    # Use only z since z_u is unobserved variable
    model = sm.OLS.from_formula(
        "delta_Y ~ D + I((1-D) * S_i) + I(D * S_i) + z", data=df_est
    ).fit()

    # Get parameters
    tau_1 = model.params["D"]
    tau_2 = model.params["I((1 - D) * S_i)"]
    tau_3 = model.params["I(D * S_i)"]

    # Calculate exposure level distribution in treatment group (distribution of S_i being 0 or 1)
    treated_mask = df_est["D"] == 1
    treated_s_binary = s_i_binary[treated_mask]

    # Probability of S_i=0 (no treated units in neighbors)
    p_s0_given_d1 = np.mean(treated_s_binary == 0)
    # Probability of S_i=1 (treated units in neighbors)
    p_s1_given_d1 = np.mean(treated_s_binary == 1)

    # Calculate direct effects at exposure levels 0 and 1
    effect_s0 = tau_1
    effect_s1 = tau_1 + tau_3 - tau_2

    # Weighted average direct effect (ODE)
    p_s1 = p_s1_given_d1
    weighted_effect = effect_s0 * (1 - p_s1) + effect_s1 * p_s1

    # Calculate standard error (using statsmodels t_test method)
    # Linear combination corresponding to ODE: τ_1 + p_s1 * τ_3 - p_s1 * τ_2
    # Written to match model parameter names
    linear_combination = f"D + {p_s1} * I(D * S_i) - {p_s1} * I((1 - D) * S_i)"

    # Apply delta method using t_test method to calculate effect and standard error
    try:
        t_test_result = model.t_test(linear_combination)
        weighted_effect = t_test_result.effect[0]
        se_weighted = t_test_result.sd[0]
    except Exception as e:
        # If t_test fails (e.g., multicollinearity)
        warnings.warn(f"Delta method failed in Modified TWFE: {e}")
        # Return NaN on failure
        return ATTEstimateResult(estimate=np.nan, standard_error=np.nan)

    return ATTEstimateResult(estimate=weighted_effect, standard_error=se_weighted)
