"""
Inverse Probability Weighting (IPW) Estimator

Standard estimation method without interference.
Based on R package ipw_did (Sant'Anna & Zhao, 2020).
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.linear_model import LogisticRegression
from ...settings import Config
from ...utils import prepare_data_for_estimation, clip_ps
from ..common.models import ATTEstimateResult


def estimate_ipw(
    df: pd.DataFrame,
    config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
) -> ATTEstimateResult:
    """Calculate Inverse Probability Weighting (IPW) estimator

    Implementation based on R code ipw_did_panel function.

    Args:
        df: Dataframe
        config: Configuration object
        covariates: List of covariate column names
        treatment_col: Treatment variable column name

    Returns:
        ATT estimate and standard error

    Raises:
        ValueError: If treatment group has 0 units

    References:
        ipw_did (Sant'Anna & Zhao, 2020), Journal of Econometrics, Vol. 219 (1)
    """
    # Use covariates
    X = df[covariates]

    D_series = df[treatment_col]
    _, delta_Y = prepare_data_for_estimation(df)

    # Prepare data
    n = len(df)
    X_matrix = np.column_stack([np.ones(n), X])  # Add intercept term
    D = D_series.values
    # delta_Y is already a numpy array from prepare_data_for_estimation()

    # Calculate propensity score (logistic regression)
    ps_model = LogisticRegression(fit_intercept=False, max_iter=1000)
    ps_model.fit(X_matrix, D)
    ps = clip_ps(ps_model.predict_proba(X_matrix)[:, 1], config)

    # Calculate IPW estimator (same method as R code)
    # Calculate weights
    w_treat = D  # Treatment group has weight 1
    w_cont = (1 - D) * ps / (1 - ps)  # Control group has IPW weights

    # Mean effect for each group
    mean_D = np.mean(D)
    if mean_D > 0:
        eta_treat = np.mean(w_treat * delta_Y) / mean_D
        eta_cont = np.mean(w_cont * delta_Y) / mean_D
    else:
        # If treatment group has 0 units
        eta_treat = 0.0
        eta_cont = 0.0

    # ATT estimate
    att = eta_treat - eta_cont

    # Calculate complete influence function (based on R code)
    # Calculate complete influence function for IPW estimator
    # Based on influence function calculation in R code ipw_did_panel function

    # 1. Calculate main term (without estimation effect)
    att_treat = w_treat * delta_Y
    att_cont = w_cont * delta_Y
    att_lin1 = att_treat - att_cont

    # 2. Calculate estimation effect of propensity score parameters
    # Calculate score function
    score_ps = (D - ps)[:, np.newaxis] * X_matrix

    # Calculate Hessian matrix (same as R code)
    W = ps * (1 - ps)
    XWX = X_matrix.T @ (W[:, np.newaxis] * X_matrix)
    try:
        Hessian_ps = np.linalg.inv(XWX) * n
    except np.linalg.LinAlgError:
        # Use pseudo-inverse for singular matrix
        Hessian_ps = np.linalg.pinv(XWX) * n

    # Calculate asymptotic linear representation
    asy_lin_rep_ps = score_ps @ Hessian_ps

    # Calculate moment function
    mom_logit = att_cont[:, np.newaxis] * X_matrix
    mom_logit = np.mean(mom_logit, axis=0)

    # Calculate estimation effect term
    att_lin2 = asy_lin_rep_ps @ mom_logit

    # 3. Calculate complete influence function
    # R code: att.inf.func <- (att.lin1 - att.lin2 - i.weights * D * ipw.att)/mean(i.weights * D)
    mean_D = np.mean(D)
    if mean_D > 0:
        influence_func = (att_lin1 - att_lin2 - D * att) / mean_D
    else:
        raise ValueError("Cannot estimate when treatment group has 0 units")

    # Calculate standard error (same formula as R code)
    se = np.std(influence_func) * np.sqrt(n - 1) / n

    return ATTEstimateResult(estimate=att, standard_error=se)
