"""
Xu Estimator: Doubly Robust (DR) Estimator

Main function for computing Xu (2025) DR estimator (DATT(g) and ODE) with HAC standard errors.
"""

import pandas as pd
import numpy as np
from ...settings import Config
from ...utils import (
    prepare_data_for_estimation,
    get_ml_model,
    define_xu_exposure_group,
    estimate_unified_outcome_model,
)
from ..common.hac import estimate_hac_se
from ..common.models import XuEstimateResult
from .propensity_scores import estimate_xu_propensity_scores
from .dr_weights import compute_xu_dr_weights
from .dr_influence import compute_xu_dr_influence_function
from typing import List, Optional


def compute_xu_dr_influence_function_ode(
    df: pd.DataFrame,
    neighbors_list: list,
    exposure_type: str,
    config: Config,
    dgp_config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Calculate ODE influence function for Xu DR estimator (similar interface to proposed methods)

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        exposure_type: Exposure type ("cs", "mo", "fm")
        config: Configuration object
        dgp_config: DGP configuration object
        covariates: List of covariate column names (default: ["z"])
        treatment_col: Treatment variable column name (default: "D")
        random_seed: Optional random seed for reproducible Xu (MO) exposure mapping

    Returns:
        Array of influence functions for ODE
    """
    df_est, delta_Y = prepare_data_for_estimation(df)
    X, D = df_est[covariates], df_est[treatment_col]
    n = len(df_est)

    # Create RNG from seed if provided (for reproducible MO exposure mapping)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    # 1. Define Exposure Mapping
    G = define_xu_exposure_group(
        df_est, neighbors_list, exposure_type, dgp_config, rng=rng
    )

    # 2. Estimate propensity score models
    ps_results = estimate_xu_propensity_scores(X, D, G, config)
    eta, eta_g, g_classes = (
        ps_results.eta,
        ps_results.eta_g,
        ps_results.g_classes,
    )
    ps_d_model = ps_results.ps_d_model
    ps_g_model = ps_results.ps_g_model

    # 3. Estimate outcome model (unified approach)
    outcome_predictions = estimate_unified_outcome_model(
        D=D.values,
        G=G.values,
        X=X.values if isinstance(X, pd.DataFrame) else X,
        delta_Y=delta_Y,
        config=config,
    )

    # 4. Calculate influence function for each g level
    influence_funcs = {}
    for g_val in g_classes:
        n = len(df_est)

        # Predictions for treatment group (D=1, G=g)
        m_1g_pred = outcome_predictions.get(f"m_delta_1{g_val}")
        if m_1g_pred is None:
            mask_1g = (D == 1) & (G == g_val)
            mean_1g = delta_Y[mask_1g].mean() if mask_1g.sum() > 0 else 0.0
            m_1g_pred = np.full(n, mean_1g)

        # Predictions for control group (D=0, G=g)
        m_0g_pred = outcome_predictions.get(f"m_delta_0{g_val}")
        if m_0g_pred is None:
            mask_0g = (D == 0) & (G == g_val)
            mean_0g = delta_Y[mask_0g].mean() if mask_0g.sum() > 0 else 0.0
            m_0g_pred = np.full(n, mean_0g)

        # Get propensity scores corresponding to exposure level g
        eta_1g_array = eta_g.get(1, {}).get(g_val, None)
        if eta_1g_array is None:
            eta_1g_array = np.full(n, config.epsilon)
        eta_0g_array = eta_g.get(0, {}).get(g_val, None)
        if eta_0g_array is None:
            eta_0g_array = np.full(n, config.epsilon)

        # Use common DR weight calculation function
        weights_for_delta_Y, adjustment = compute_xu_dr_weights(
            D=D.values,
            G=G.values,
            eta=eta,
            eta_1g=eta_1g_array,
            eta_0g=eta_0g_array,
            m_1g_pred=m_1g_pred,
            m_0g_pred=m_0g_pred,
            g_level=g_val,
        )

        # DATT estimator: normalize by number of treated units
        N_treated = D.sum()
        if N_treated > 0:
            datt_g = np.sum(weights_for_delta_Y * delta_Y + adjustment) / N_treated
        else:
            datt_g = 0.0

        # Calculate influence function
        influence_func = compute_xu_dr_influence_function(
            D=D.values,
            G=G.values,
            delta_Y=delta_Y,
            eta=eta,
            eta_1g=eta_1g_array,
            eta_0g=eta_0g_array,
            m_1g_pred=m_1g_pred,
            m_0g_pred=m_0g_pred,
            g_level=g_val,
            X=X,
            ps_d_model=ps_d_model,
            ps_g_model=ps_g_model,
            outcome_model_1g=None,  # Unified approach uses predictions directly
            outcome_model_0g=None,  # Unified approach uses predictions directly
            n=n,
            datt_g=datt_g,
        )
        influence_funcs[g_val] = influence_func

    # 5. Calculate ODE influence function (weighted average)
    p_g_d1 = G[D == 1].value_counts(normalize=True)
    influence_ode = np.zeros(n)
    for g in g_classes:
        if g in influence_funcs:
            influence_ode += influence_funcs[g] * p_g_d1.get(g, 0)

    return influence_ode


def compute_xu_dr_influence_function_ode_no_adjustment(
    df: pd.DataFrame,
    neighbors_list: list,
    exposure_type: str,
    config: Config,
    dgp_config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Calculate ODE influence function for Xu DR estimator (no adjustment version)

    Calculates with adjustment term set to 0 to verify whether DR estimator
    behaves similarly to IPW estimator.

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        exposure_type: Exposure type ("cs", "mo", "fm")
        config: Configuration object
        dgp_config: DGP configuration object
        covariates: List of covariate column names (default: ["z"])
        treatment_col: Treatment variable column name (default: "D")
        random_seed: Optional random seed for reproducible Xu (MO) exposure mapping

    Returns:
        Array of influence functions for ODE (no adjustment)
    """
    df_est, delta_Y = prepare_data_for_estimation(df)
    X, D = df_est[covariates], df_est[treatment_col]
    n = len(df_est)

    # Create RNG from seed if provided (for reproducible MO exposure mapping)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    # 1. Define Exposure Mapping
    G = define_xu_exposure_group(
        df_est, neighbors_list, exposure_type, dgp_config, rng=rng
    )

    # 2. Estimate propensity score models
    ps_results = estimate_xu_propensity_scores(X, D, G, config)
    eta, eta_g, g_classes = (
        ps_results.eta,
        ps_results.eta_g,
        ps_results.g_classes,
    )
    ps_d_model = ps_results.ps_d_model
    ps_g_model = ps_results.ps_g_model

    # 3. Estimate outcome model (unified approach)
    outcome_predictions = estimate_unified_outcome_model(
        D=D.values,
        G=G.values,
        X=X.values if isinstance(X, pd.DataFrame) else X,
        delta_Y=delta_Y,
        config=config,
    )

    # 4. Calculate influence function for each g level (no adjustment)
    influence_funcs = {}
    for g_val in g_classes:
        n = len(df_est)

        # Get propensity scores corresponding to exposure level g
        eta_1g_array = eta_g.get(1, {}).get(g_val, None)
        if eta_1g_array is None:
            eta_1g_array = np.full(n, config.epsilon)
        eta_0g_array = eta_g.get(0, {}).get(g_val, None)
        if eta_0g_array is None:
            eta_0g_array = np.full(n, config.epsilon)

        # Calculate predictions even without adjustment (unified approach)
        m_1g_pred = outcome_predictions.get(f"m_delta_1{g_val}")
        if m_1g_pred is None:
            m_1g_pred = np.zeros(n)
        m_0g_pred = outcome_predictions.get(f"m_delta_0{g_val}")
        if m_0g_pred is None:
            m_0g_pred = np.zeros(n)

        # Use common DR weight calculation function (adjustment is calculated but set to 0 later)
        weights_for_delta_Y, _ = compute_xu_dr_weights(
            D=D.values,
            G=G.values,
            eta=eta,
            eta_1g=eta_1g_array,
            eta_0g=eta_0g_array,
            m_1g_pred=m_1g_pred,
            m_0g_pred=m_0g_pred,
            g_level=g_val,
        )

        # No adjustment: adjustment = 0
        adjustment = np.zeros(n)

        # DATT estimator: normalize by number of treated units (no adjustment)
        N_treated = D.sum()
        if N_treated > 0:
            datt_g = np.sum(weights_for_delta_Y * delta_Y + adjustment) / N_treated
        else:
            datt_g = 0.0

        # Calculate influence function (no adjustment)
        # weighted_delta = weights_for_delta_Y * delta_Y + 0
        weighted_delta = weights_for_delta_Y * delta_Y + adjustment
        # For DATT estimator, centering should subtract datt_g only from treatment group (D=1)
        tau_lin1 = weighted_delta - (datt_g * D.values)
        influence_func = tau_lin1

        influence_funcs[g_val] = influence_func

    # 5. Calculate ODE influence function (weighted average)
    p_g_d1 = G[D == 1].value_counts(normalize=True)
    influence_ode = np.zeros(n)
    for g in g_classes:
        if g in influence_funcs:
            influence_ode += influence_funcs[g] * p_g_d1.get(g, 0)

    return influence_ode


def estimate_xu_dr(
    df: pd.DataFrame,
    neighbors_list: list,
    exposure_type: str,
    config: Config,
    dgp_config: Config,
    locations: np.ndarray,
    K: float,
    random_seed: Optional[int] = None,
) -> XuEstimateResult:
    """Calculate Xu (2025) Doubly Robust (DR) estimator (for DATT)

    Estimates direct effect DATT(g) (Direct Average Treatment Effect on the Treated)
    for each exposure level and overall direct effect ODE (Overall Direct Effect).

    DATT estimator is normalized by the number of treated units because it estimates effects for the treated group.
    Weight calculation is performed with treated group=1, control group=odds ratio.

    Args:
        df: DataFrame
        neighbors_list: Neighbors list
        exposure_type: Exposure type ("cs", "mo", "fm")
        config: Configuration object
        dgp_config: DGP configuration object
        locations: Spatial position array
        K: Bandwidth parameter
        random_seed: Optional random seed for reproducible Xu (MO) exposure mapping

    Returns:
        DATT(g) estimate, ODE, standard errors

    References:
        Xu, R. (2025). "Difference-in-Differences with Interference"
    """
    df_est, delta_Y = prepare_data_for_estimation(df)
    # z_u is unobserved variable, so use only z
    X, D = df_est[["z"]], df_est["D"]
    n = len(df_est)

    # Create RNG from seed if provided (for reproducible MO exposure mapping)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    # 1. Define Exposure Mapping (reuse existing helper function)
    G = define_xu_exposure_group(
        df_est, neighbors_list, exposure_type, dgp_config, rng=rng
    )

    # 2. Estimate propensity score model (reuse existing helper function)
    ps_results = estimate_xu_propensity_scores(X, D, G, config)
    eta, eta_g, g_classes = (
        ps_results.eta,
        ps_results.eta_g,
        ps_results.g_classes,
    )
    ps_d_model = ps_results.ps_d_model
    ps_g_model = ps_results.ps_g_model

    # 3. Estimate outcome model (unified approach)
    outcome_predictions = estimate_unified_outcome_model(
        D=D.values,
        G=G.values,
        X=X.values if isinstance(X, pd.DataFrame) else X,
        delta_Y=delta_Y,
        config=config,
    )

    # 4. Calculate doubly robust estimator (DATT(g) for each g)
    dr_ests = {}
    dr_se = {}
    influence_funcs = {}

    for g_val in g_classes:
        # Calculate vectorized predictions (predict for all data X at once)
        n = len(df_est)

        # Predicted values for treated group (D=1, G=g)
        m_1g_pred = outcome_predictions.get(f"m_delta_1{g_val}")
        if m_1g_pred is None:
            # If no predicted values, calculate mean value for corresponding group and create prediction array
            mask_1g = (D == 1) & (G == g_val)
            mean_1g = delta_Y[mask_1g].mean() if mask_1g.sum() > 0 else 0.0
            m_1g_pred = np.full(n, mean_1g)

        # Predictions for control group (D=0, G=g)
        m_0g_pred = outcome_predictions.get(f"m_delta_0{g_val}")
        if m_0g_pred is None:
            # If predictions not available, calculate mean of corresponding group and create prediction array
            mask_0g = (D == 0) & (G == g_val)
            mean_0g = delta_Y[mask_0g].mean() if mask_0g.sum() > 0 else 0.0
            m_0g_pred = np.full(n, mean_0g)

        # Get propensity scores corresponding to exposure level g (as array)
        eta_1g_array = eta_g.get(1, {}).get(g_val, None)
        if eta_1g_array is None:
            eta_1g_array = np.full(n, config.epsilon)
        eta_0g_array = eta_g.get(0, {}).get(g_val, None)
        if eta_0g_array is None:
            eta_0g_array = np.full(n, config.epsilon)

        # Use common DR weight calculation function
        weights_for_delta_Y, adjustment = compute_xu_dr_weights(
            D=D.values,
            G=G.values,
            eta=eta,  # eta is already a numpy array from clip_ps()
            eta_1g=eta_1g_array,
            eta_0g=eta_0g_array,
            m_1g_pred=m_1g_pred,
            m_0g_pred=m_0g_pred,
            g_level=g_val,
        )

        # DATT estimator: normalize by number of treated units
        N_treated = D.sum()
        if N_treated > 0:
            datt_g = np.sum(weights_for_delta_Y * delta_Y + adjustment) / N_treated
        else:
            datt_g = 0.0
        dr_ests[f"datt_g{g_val}"] = datt_g

        # Calculate influence function
        influence_func = compute_xu_dr_influence_function(
            D=D.values,
            G=G.values,
            delta_Y=delta_Y,
            eta=eta,
            eta_1g=eta_1g_array,
            eta_0g=eta_0g_array,
            m_1g_pred=m_1g_pred,
            m_0g_pred=m_0g_pred,
            g_level=g_val,
            X=X,
            ps_d_model=ps_d_model,
            ps_g_model=ps_g_model,
            outcome_model_1g=None,  # Unified approach uses predictions directly
            outcome_model_0g=None,  # Unified approach uses predictions directly
            n=n,
            datt_g=datt_g,
        )
        influence_funcs[g_val] = influence_func

        # Calculate HAC standard error
        hac_se = estimate_hac_se(influence_func, locations, K, config=config)
        dr_se[f"datt_g{g_val}_se"] = hac_se

    # 5. Calculate Overall Direct Effect (ODE)
    # Weighted average of DATT(g) by distribution of G in treatment group
    p_g_d1 = G[D == 1].value_counts(normalize=True)
    ode = sum(dr_ests.get(f"datt_g{g}") * p_g_d1.get(g, 0) for g in g_classes)

    # Calculate ODE influence function (weighted average)
    influence_ode = np.zeros(n)
    for g in g_classes:
        if g in influence_funcs:
            influence_ode += influence_funcs[g] * p_g_d1.get(g, 0)

    # Calculate HAC standard error
    ode_se = estimate_hac_se(influence_ode, locations, K, config=config)

    return XuEstimateResult(
        datt_estimates=dr_ests,
        ode=ode,
        datt_standard_errors=dr_se,
        ode_standard_error=ode_se,
    )
