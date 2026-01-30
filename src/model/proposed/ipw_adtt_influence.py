"""
Proposed Method: IPW ADTT Influence Function Computation

Computes the influence function for the proposed IPW ADTT (Average Direct Treatment on Treated) estimator.
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


def compute_adtt_influence_function(
    df: pd.DataFrame,
    neighbors_list: List[list],
    model_type: str,
    config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
) -> np.ndarray:
    """Calculate influence function for proposed ADTT estimation method

    Based on definition in paper draft_v2.tex Section 3.1:

    Z_i^{ADTT} = (D_i - e_i) / (π_i(1 - e_i)) (Y_2i - Y_1i)

    where e_i = Pr(D_i=1 | Z_i, D_{N_A(i;K)}), π_i = Pr(D_i=1 | Z_i)

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

    # Calculate influence function: (D_i - e_i) / (π_i(1 - e_i)) * (Y_2i - Y_1i)
    return ((D - ps) / (pi_i * (1 - ps))) * delta_Y
