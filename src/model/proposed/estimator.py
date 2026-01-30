"""
Proposed Method: Estimator with Standard Error

Common wrapper for proposed methods (ADTT/AITT) that computes estimates with HAC standard errors.
"""

import numpy as np
import pandas as pd
from typing import Callable, Literal
from ...settings import Config
from ..common.hac import estimate_hac_se
from ..common.models import ATTEstimateResult


def estimate_proposed_with_se(
    df: pd.DataFrame,
    neighbors_list: list,
    locations: np.ndarray,
    K: float,
    influence_calculator: Callable,
    model_type: Literal["logistic"],
    config: Config,
) -> ATTEstimateResult:
    """Common wrapper for proposed methods (ADTT/AITT) estimation process

    Calculates influence function, uses its mean as ATT estimate, and calculates HAC standard error.

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        locations: Spatial location array
        K: Bandwidth parameter
        influence_calculator: Influence function calculation function
        model_type: Model type
        config: Configuration object

    Returns:
        ATT estimate and HAC standard error
    """
    influence_func = influence_calculator(df, neighbors_list, model_type, config)

    estimate = np.mean(influence_func)
    # Calculate HAC standard error
    hac_se = estimate_hac_se(influence_func, locations, K, config=config)

    return ATTEstimateResult(estimate=estimate, standard_error=hac_se)
