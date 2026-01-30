"""
Two-Way Fixed Effects (TWFE) Estimator

Standard estimation method without interference.
"""

import pandas as pd
import statsmodels.api as sm
from typing import List
from ...utils import prepare_data_for_estimation
from ..common.models import ATTEstimateResult


def estimate_twfe(
    df: pd.DataFrame,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
) -> ATTEstimateResult:
    """Calculate Two-Way Fixed Effects (TWFE) estimator

    Model formula: delta_Y = β_0 + τ * D + β_1' * Z + ε

    Args:
        df: Dataframe
        covariates: List of covariate column names
        treatment_col: Treatment variable column name

    Returns:
        ATT estimate and standard error
    """
    df_est, _ = prepare_data_for_estimation(df)

    # Join multiple covariates with "+"
    covariate_str = " + ".join(covariates)

    # Dynamically generate formula string
    formula = f"delta_Y ~ {treatment_col} + {covariate_str}"
    model = sm.OLS.from_formula(formula, data=df_est).fit()

    return ATTEstimateResult(
        estimate=model.params[treatment_col], standard_error=model.bse[treatment_col]
    )
