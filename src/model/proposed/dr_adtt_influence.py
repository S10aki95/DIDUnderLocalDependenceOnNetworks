"""
Proposed Method: DR ADTT Influence Function Computation

Computes the doubly robust (DR) influence function for the proposed ADTT estimator.
This combines IPW with outcome regression models for double robustness.
"""

import pandas as pd
import numpy as np
from typing import List
from ...settings import Config
from ...utils import (
    prepare_data_for_estimation,
    get_ml_model,
    clip_ps,
    create_neighbor_features,
    estimate_outcome_model_from_features,
)


def compute_dr_adtt_influence_function(
    df: pd.DataFrame,
    neighbors_list: List[list],
    model_type: str,
    config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
) -> np.ndarray:
    """Calculate influence function for proposed DR ADTT estimation method

    Based on definition in paper draftv_ver4.tex Section 3.2:

    ADTT^DR = (1/n) Σ_i {
        (D_i/π̂_i)(ΔY_i - Δm_{1i})
        - ((1-D_i)ê_i)/(π̂_i(1-ê_i)) * (ΔY_i - Δm_0)
        + (D_i/π̂_i)(Δm_1 - Δm_0)
    }

    Influence function Z_i^{DR} (equation \ref{eq:Z_DR}):
    Z_i^{DR} = (D_i/π̂_i)(ΔY_i - Δm_{1i})
              - ((1-D_i)ê_i)/(π̂_i(1-ê_i)) * (ΔY_i - Δm_0)
              + (D_i/π̂_i)(Δm_1 - Δm_0)

    where e_i = Pr(D_i=1 | Z_i, D_{N_A(i;K)}), π_i = Pr(D_i=1 | Z_i),
    μ_1i(D_{N_A(i;K)}, z_i) = E[Y_2i - Y_1i | D_i=1, D_{N_A(i;K)}, z_i],
    μ_0i(D_{N_A(i;K)}, z_i) = E[Y_2i - Y_1i | D_i=0, D_{N_A(i;K)}, z_i],
    Δm_{1i} = μ̂_{1i}(D_{N_A(i;K)}, z_i),
    Δm_0 = μ̂_0(D_{N_A(i;K)}, z_i),
    Δm_1 = μ̂_1(D_{N_A(i;K)}, z_i),
    Δm_0 = μ̂_0(D_{N_A(i;K)}, z_i)

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        model_type: Model type
        config: Configuration object
        covariates: List of covariate column names
        treatment_col: Treatment variable column name

    Returns:
        Array of DR influence functions
    """
    D = df[treatment_col].values
    _, delta_Y = prepare_data_for_estimation(df)
    N = len(df)

    # Create neighbor features (for propensity score estimation)
    X_features_df = create_neighbor_features(
        df, neighbors_list, config, covariates=covariates, treatment_col=treatment_col
    )

    # Estimate ADTT propensity score e_i
    ps_model = get_ml_model(model_type, config).fit(X_features_df, D)
    ps = clip_ps(ps_model.predict_proba(X_features_df)[:, 1], config)

    # Estimate marginal treatment probability π_i (using only Z_i)
    Z = df[covariates].values
    pi_model = get_ml_model(model_type, config).fit(Z, D)
    pi_i = clip_ps(pi_model.predict_proba(Z)[:, 1], config)

    # Estimate outcome regression model (neighbor treatment vector based)
    # Following paper definition, condition on entire D_{N_A(i;K)}
    outcome_predictions = estimate_outcome_model_from_features(
        D=D,
        X_features_df=X_features_df,
        delta_Y=delta_Y,
        config=config,
    )

    # Get predictions
    mu_1_pred = outcome_predictions["m_delta_1"]
    mu_0_pred = outcome_predictions["m_delta_0"]

    # Fill missing predictions with mean values
    mask_d1 = D == 1
    if mask_d1.sum() > 0 and np.all(mu_1_pred[mask_d1] == 0):
        mean_d1 = delta_Y[mask_d1].mean() if mask_d1.sum() > 0 else 0.0
        mu_1_pred[mask_d1] = mean_d1

    mask_d0 = D == 0
    if mask_d0.sum() > 0 and np.all(mu_0_pred[mask_d0] == 0):
        mean_d0 = delta_Y[mask_d0].mean() if mask_d0.sum() > 0 else 0.0
        mu_0_pred[mask_d0] = mean_d0

    # Calculate DR influence function (equation \ref{eq:Z_DR})
    # Term 1: Residual term for D=1
    term1 = (D / pi_i) * (delta_Y - mu_1_pred)

    # Term 2: Residual term for D=0 (note minus sign)
    term2 = -1 * ((1 - D) * ps) / (pi_i * (1 - ps)) * (delta_Y - mu_0_pred)

    # Term 3: Difference in regression predictions (Trend) - weighted by ê_i/π_i for ADTT
    term3 = (ps / pi_i) * (mu_1_pred - mu_0_pred)

    return term1 + term2 + term3
