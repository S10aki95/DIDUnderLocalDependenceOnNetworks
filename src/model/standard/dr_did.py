"""
Doubly Robust Difference-in-Differences (DR-DID) Estimator

Standard estimation method without interference.
Based on R package DRDID (Sant'Anna & Zhao, 2020).
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.linear_model import LogisticRegression, LinearRegression
from ...settings import Config
from ...utils import prepare_data_for_estimation, clip_ps
from ..common.models import ATTEstimateResult


def estimate_dr_did(
    df: pd.DataFrame,
    config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
) -> ATTEstimateResult:
    """Calculate Doubly Robust Difference-in-Differences (DR-DID) estimator

    Implementation based on R code drdid_panel function.

    Args:
        df: Dataframe
        config: Configuration object
        covariates: List of covariate column names
        treatment_col: Treatment variable column name

    Returns:
        ATT estimate and standard error

    References:
        DRDID (Sant'Anna & Zhao, 2020), Journal of Econometrics, Vol. 219 (1)
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

    # 1. Estimate propensity score (logistic regression)
    # NOTE: Weighting not needed - don't specify sample_weight for equal weights
    # NOTE: Trimming already implemented in clip_ps function (config.ps_clip_min/max)
    ps_model = LogisticRegression(fit_intercept=False, max_iter=1000)
    ps_model.fit(X_matrix, D)
    ps = clip_ps(ps_model.predict_proba(X_matrix)[:, 1], config)

    # 2. Estimate outcome regression for control group (least squares)
    # NOTE: Weighting not needed - don't specify sample_weight for equal weights
    control_mask = D == 0
    if control_mask.sum() > 0:
        reg_model = LinearRegression(fit_intercept=False)
        reg_model.fit(X_matrix[control_mask], delta_Y[control_mask])
        out_delta = reg_model.predict(X_matrix)
    else:
        out_delta = np.zeros_like(delta_Y)

    # 3. Implement weighting structure
    # NOTE: i_weights (observation weights) are equal weights so omit 1
    # NOTE: trim_ps (trimming) already handled by clip_ps so omit
    # Treatment group uses weight 1, control group uses IPW weights
    w_treat = D  # Treatment group has weight 1
    w_cont = (1 - D) * ps / (1 - ps)  # Control group has IPW weights

    # 4. Calculate DR estimator
    dr_att_treat = w_treat * (delta_Y - out_delta)
    dr_att_cont = w_cont * (delta_Y - out_delta)

    mean_w_treat = np.mean(w_treat)
    mean_w_cont = np.mean(w_cont)

    if mean_w_treat > 0 and mean_w_cont > 0:
        eta_treat = np.mean(dr_att_treat) / mean_w_treat
        eta_cont = np.mean(dr_att_cont) / mean_w_cont
    else:
        eta_treat = 0.0
        eta_cont = 0.0

    dr_att = eta_treat - eta_cont

    # 5. Calculate complete influence function
    # Calculate complete influence function for DR-DID estimator
    # Based on influence function calculation in R code drdid_panel function

    # 1. Asymptotic linear representation of outcome regression (R code lines 156-166)
    # NOTE: i_weights (observation weights) are equal weights so omit 1
    weights_ols = 1 - D
    wols_x = weights_ols[:, np.newaxis] * X_matrix
    wols_eX = (
        weights_ols[:, np.newaxis] * (delta_Y - out_delta)[:, np.newaxis] * X_matrix
    )

    # Calculate X'X matrix
    XpX = (wols_x.T @ X_matrix) / n

    # Check for singular matrix and calculate inverse
    # Use pseudo-inverse if condition number is large (nearly singular) or matrix is singular
    try:
        # Use pseudo-inverse if NaN or Inf included
        if np.any(np.isnan(XpX)) or np.any(np.isinf(XpX)):
            XpX_inv = np.linalg.pinv(XpX)
        else:
            cond_num = np.linalg.cond(XpX)
            if cond_num > 1e12 or cond_num < 1e-12:
                # Use pseudo-inverse if condition number too large or too small
                XpX_inv = np.linalg.pinv(XpX)
            else:
                XpX_inv = np.linalg.inv(XpX)
    except (np.linalg.LinAlgError, ValueError):
        # Use pseudo-inverse for singular matrix or other errors
        XpX_inv = np.linalg.pinv(XpX)

    # Calculate asymptotic linear representation
    asy_lin_rep_wols = wols_eX @ XpX_inv

    # 2. Asymptotic linear representation of propensity score (R code lines 168-172)
    # NOTE: i_weights (observation weights) are equal weights so omit 1
    score_ps = (D - ps)[:, np.newaxis] * X_matrix
    W = ps * (1 - ps)
    XWX = X_matrix.T @ (W[:, np.newaxis] * X_matrix)

    # Check for singular matrix and calculate inverse
    # Use pseudo-inverse if condition number is large (nearly singular) or matrix is singular
    try:
        cond_num = np.linalg.cond(XWX)
        if (
            cond_num > 1e12
            or cond_num < 1e-12
            or np.any(np.isnan(XWX))
            or np.any(np.isinf(XWX))
        ):
            # Use pseudo-inverse if condition number too large or too small, or NaN/Inf included
            Hessian_ps = np.linalg.pinv(XWX) * n
        else:
            Hessian_ps = np.linalg.inv(XWX) * n
    except (np.linalg.LinAlgError, ValueError):
        # Use pseudo-inverse for singular matrix or other errors
        Hessian_ps = np.linalg.pinv(XWX) * n

    asy_lin_rep_ps = score_ps @ Hessian_ps

    # 3. Influence function for treatment group (R code lines 175-186)
    # Main term
    inf_treat_1 = dr_att_treat - w_treat * eta_treat

    # Estimation effect (outcome regression)
    M1 = np.mean(w_treat[:, np.newaxis] * X_matrix, axis=0)
    inf_treat_2 = asy_lin_rep_wols @ M1

    # Influence function for treatment group
    mean_w_treat = np.mean(w_treat)
    if mean_w_treat > 0:
        inf_treat = (inf_treat_1 - inf_treat_2) / mean_w_treat
    else:
        inf_treat = np.zeros_like(inf_treat_1)

    # 4. Influence function for control group (R code lines 188-213)
    # Main term
    inf_cont_1 = dr_att_cont - w_cont * eta_cont

    # Estimation effect (propensity score)
    M2 = np.mean(
        w_cont[:, np.newaxis]
        * (delta_Y - out_delta - eta_cont)[:, np.newaxis]
        * X_matrix,
        axis=0,
    )
    inf_cont_2 = asy_lin_rep_ps @ M2

    # Estimation effect (outcome regression)
    M3 = np.mean(w_cont[:, np.newaxis] * X_matrix, axis=0)
    inf_cont_3 = asy_lin_rep_wols @ M3

    # Influence function for control group
    mean_w_cont = np.mean(w_cont)
    if mean_w_cont > 0:
        inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / mean_w_cont
    else:
        inf_control = np.zeros_like(inf_cont_1)

    # 5. Final influence function (R code line 216)
    influence_func = inf_treat - inf_control

    # 6. Calculate standard error (same formula as R code)
    se = np.std(influence_func) * np.sqrt(n - 1) / n

    return ATTEstimateResult(estimate=dr_att, standard_error=se)
