"""
Proposed Method: IPW AITT Influence Function Computation

Computes the influence function for the proposed IPW AITT (Average Indirect Treatment on Treated) estimator.
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
)


def compute_aitt_influence_function(
    df: pd.DataFrame,
    neighbors_list: List[list],
    model_type: str,
    config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
    random_seed: Optional[int] = None,
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
        random_seed: Optional random seed for reproducible neighbor sampling

    Returns:
        Array of influence functions
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

    # Generate features for all pairs at once
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
