"""
Proposed Method: DR AITT Influence Function Computation

Computes the doubly robust (DR) influence function for the proposed AITT estimator.
This combines IPW with outcome regression models for double robustness.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from ...settings import Config
from ...utils import (
    prepare_data_for_estimation,
    get_ml_model,
    clip_ps,
    create_neighbor_features,
    estimate_outcome_model_from_features,
)


def compute_dr_aitt_influence_function(
    df: pd.DataFrame,
    neighbors_list: List[list],
    model_type: str,
    config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Calculate influence function for proposed DR AITT estimation method

    Based on definition in paper draftv_ver4.tex Section 3.2:

    AITT^DR = (1/n) Σ_i {
        (1/|N_A(i;K)|) Σ_{j ∈ N_A(i;K)} {
            (D_i/π̂_i)(ΔY_j - Δm'_{1,ij})
            - ((1-D_i)ê'_{ij})/(π̂_i(1-ê'_{ij})) * (ΔY_j - Δm'_{0,ij})
            + (D_i/π̂_i)(Δm'_{1,ij} - Δm'_{0,ij})
        }
    }

    Influence function Z_i^{DR} (equation \ref{eq:Z_DR}):
    Z_i^{DR} = (1/|N_A(i;K)|) Σ_j {
        (D_i/π̂_i)(ΔY_j - Δm'_{1,ij})
        - ((1-D_i)ê'_{ij})/(π̂_i(1-ê'_{ij})) * (ΔY_j - Δm'_{0,ij})
        + (D_i/π̂_i)(Δm'_{1,ij} - Δm'_{0,ij})
    }

    where e'_ij = Pr(D_i=1 | Z_i, D_j, D_{N_A(j;K)}^{-i}), π_i = Pr(D_i=1 | Z_i),
    μ_1ij(D_j, D_{N_A(j;K)}^{-i}, z_i, z_j) = E[Y_2j - Y_1j | D_i=1, D_j, D_{N_A(j;K)}^{-i}, z_i, z_j],
    μ_0ij(D_j, D_{N_A(j;K)}^{-i}, z_i, z_j) = E[Y_2j - Y_1j | D_i=0, D_j, D_{N_A(j;K)}^{-i}, z_i, z_j],
    Δm'_{1,ij} = μ̂_{1ij}(D_j, D_{N_A(j;K)}^{-i}, z_i, z_j),
    Δm'_{0,ij} = μ̂_{0ij}(D_j, D_{N_A(j;K)}^{-i}, z_i, z_j)

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        model_type: Model type
        config: Configuration object
        covariates: List of covariate column names
        treatment_col: Treatment variable column name
        random_seed: Optional random seed for reproducible neighbor sampling

    Returns:
        Array of DR influence functions
    """
    D = df[treatment_col].values
    _, delta_Y = prepare_data_for_estimation(df)
    N = len(df)

    # Read value from config
    max_neighbors = config.max_neighbors

    # Create random number generator for neighbor sampling
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    # Create pair data (i, j) for AITT estimation
    # When number of neighbors exceeds L, randomly sample L neighbors to avoid arbitrary selection
    pairs = []
    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) > max_neighbors:
            # Randomly sample max_neighbors neighbors
            if rng is not None:
                neighbors_array = np.array(neighbors)
                selected_indices = rng.choice(
                    len(neighbors_array), size=max_neighbors, replace=False
                )
                selected_neighbors = neighbors_array[selected_indices].tolist()
            else:
                # Fallback to first L if no rng (backward compatibility)
                selected_neighbors = neighbors[:max_neighbors]
            pairs.extend([(i, j) for j in selected_neighbors])
        else:
            pairs.extend([(i, j) for j in neighbors])

    if len(pairs) == 0:
        # Return zero array if no pairs
        return np.zeros(N)

    # Generate features for all pairs at once (for propensity score estimation)
    X_pairs_df = create_neighbor_features(
        df,
        neighbors_list,
        config,
        for_aitt=True,
        pairs=pairs,
        covariates=covariates,
        treatment_col=treatment_col,
        rng=rng,
    )
    y_pairs = D[[i for i, j in pairs]]

    # Estimate AITT propensity score e'_ij
    ps_model = get_ml_model(model_type, config).fit(X_pairs_df, y_pairs)
    ps_prime = clip_ps(ps_model.predict_proba(X_pairs_df)[:, 1], config)

    # Estimate marginal treatment probability π_i (using only Z_i)
    Z = df[covariates].values
    pi_model = get_ml_model(model_type, config).fit(Z, D)
    pi_i = clip_ps(pi_model.predict_proba(Z)[:, 1], config)

    # Create DataFrame for pair data (used in outcome regression model and influence function calculation)
    df_pairs = pd.DataFrame(pairs, columns=["i", "j"])
    df_pairs["delta_Y_j"] = delta_Y[df_pairs["j"]]
    df_pairs["D_i"] = y_pairs
    df_pairs["D_j"] = D[df_pairs["j"]]

    # Estimate outcome regression model (neighbor treatment vector based)
    # Following paper definition, condition on entire D_j, D_{N_A(j;K)}^{-i}
    outcome_predictions = estimate_outcome_model_from_features(
        D=y_pairs,
        X_features_df=X_pairs_df,
        delta_Y=df_pairs["delta_Y_j"].values,
        config=config,
    )

    # Get predictions (for D_i=1 and D_i=0 cases)
    mu_1ij_pred = outcome_predictions["m_delta_1"]
    mu_0ij_pred = outcome_predictions["m_delta_0"]

    # Fill missing predictions with mean values
    mask_di1 = df_pairs["D_i"] == 1
    if mask_di1.sum() > 0:
        if np.all(mu_1ij_pred[mask_di1] == 0):
            mean_di1 = (
                df_pairs.loc[mask_di1, "delta_Y_j"].mean()
                if mask_di1.sum() > 0
                else 0.0
            )
            mu_1ij_pred[mask_di1] = mean_di1

    mask_di0 = df_pairs["D_i"] == 0
    if mask_di0.sum() > 0:
        if np.all(mu_0ij_pred[mask_di0] == 0):
            mean_di0 = (
                df_pairs.loc[mask_di0, "delta_Y_j"].mean()
                if mask_di0.sum() > 0
                else 0.0
            )
            mu_0ij_pred[mask_di0] = mean_di0

    # Add additional data to df_pairs for influence function calculation
    df_pairs["pi_i"] = pi_i[df_pairs["i"]]
    df_pairs["ps_prime"] = ps_prime
    df_pairs["mu_1ij_pred"] = mu_1ij_pred
    df_pairs["mu_0ij_pred"] = mu_0ij_pred

    # Calculate DR influence function (for each pair, equation \ref{eq:Z_DR})
    # Term 1: Residual term for D_i=1
    df_pairs["term1"] = (df_pairs["D_i"] / df_pairs["pi_i"]) * (
        df_pairs["delta_Y_j"] - df_pairs["mu_1ij_pred"]
    )

    # Term 2: Residual term for D_i=0 (note minus sign)
    df_pairs["term2"] = (
        -1
        * ((1 - df_pairs["D_i"]) * df_pairs["ps_prime"])
        / (df_pairs["pi_i"] * (1 - df_pairs["ps_prime"]))
        * (df_pairs["delta_Y_j"] - df_pairs["mu_0ij_pred"])
    )

    # Term 3: Difference in regression predictions (Trend) - weighted by ê'_{ij}/π_i for AITT
    df_pairs["term3"] = (df_pairs["ps_prime"] / df_pairs["pi_i"]) * (
        df_pairs["mu_1ij_pred"] - df_pairs["mu_0ij_pred"]
    )

    # DR influence function = Term 1 + Term 2 + Term 3
    df_pairs["dr_influence"] = df_pairs["term1"] + df_pairs["term2"] + df_pairs["term3"]

    # Average over neighbors j for each i to get influence function
    influence_func_series = df_pairs.groupby("i")["dr_influence"].mean()
    return influence_func_series.reindex(range(N), fill_value=0).values
