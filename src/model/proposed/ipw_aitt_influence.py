"""
Proposed Method: IPW AITT Influence Function Computation

Computes the influence function for the proposed IPW AITT (Average Indirect Treatment on Treated) estimator.
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
)


def compute_aitt_influence_function(
    df: pd.DataFrame,
    neighbors_list: List[list],
    model_type: str,
    config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
) -> np.ndarray:
    """Calculate influence function for proposed AITT estimation method

    Based on definition in paper draft_v2.tex Section 3.1:

    Z_i^{AITT} = (1/|N_A(i;K)|) Σ_j (D_i - e'_ij) / (π_i(1 - e'_ij)) (Y_2j - Y_1j)

    where e'_ij = Pr(D_i=1 | Z_i, D_j, D_{N_A(j;K)}^{-i}), π_i = Pr(D_i=1 | Z_i)

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        model_type: Model type
        config: Configuration object
        covariates: List of covariate column names
        treatment_col: Treatment variable column name

    Returns:
        Array of influence functions
    """
    D = df[treatment_col].values
    _, delta_Y = prepare_data_for_estimation(df)
    N = len(df)

    # Read value from config
    max_neighbors_per_unit = config.max_neighbors_per_unit

    # Create pair data (i, j) for AITT estimation
    # Following paper description, take top L neighbors from distance-sorted neighbors
    # neighbors_list is already sorted by distance in find_neighbors function
    pairs = []
    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) > max_neighbors_per_unit:
            # Already sorted by distance, so take top max_neighbors_per_unit
            selected_neighbors = neighbors[:max_neighbors_per_unit]
            pairs.extend([(i, j) for j in selected_neighbors])
        else:
            pairs.extend([(i, j) for j in neighbors])

    # Generate features for all pairs at once
    X_pairs_df = create_neighbor_features(
        df,
        neighbors_list,
        config,
        for_aitt=True,
        pairs=pairs,
        covariates=covariates,
        treatment_col=treatment_col,
    )
    y_pairs = D[[i for i, j in pairs]]

    # Estimate AITT propensity score e'_ij
    ps_model = get_ml_model(model_type, config).fit(X_pairs_df, y_pairs)
    ps_prime = clip_ps(ps_model.predict_proba(X_pairs_df)[:, 1], config)

    # Estimate marginal treatment probability π_i (using only Z_i)
    Z = df[covariates].values

    pi_model = get_ml_model(model_type, config).fit(Z, D)
    pi_i = clip_ps(pi_model.predict_proba(Z)[:, 1], config)

    df_pairs = pd.DataFrame(pairs, columns=["i", "j"])
    df_pairs["delta_Y_j"] = delta_Y[df_pairs["j"]]
    df_pairs["pi_i"] = pi_i[df_pairs["i"]]

    # Calculate influence function: (D_i - e'_ij) / (π_i(1 - e'_ij)) * (Y_2j - Y_1j)
    df_pairs["ipw_term"] = (
        (y_pairs - ps_prime) / (df_pairs["pi_i"] * (1 - ps_prime))
    ) * df_pairs["delta_Y_j"]

    # Average over neighbors j for each i to get influence function
    influence_func_series = df_pairs.groupby("i")["ipw_term"].mean()
    return influence_func_series.reindex(range(N), fill_value=0).values
