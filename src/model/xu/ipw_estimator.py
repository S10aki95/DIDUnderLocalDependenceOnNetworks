"""
Xu Estimator: IPW Estimator

Main function for computing Xu (2025) IPW estimator (DATT(g) and ODE) with HAC standard errors.
"""

import pandas as pd
import numpy as np
from ...settings import Config
from ...utils import prepare_data_for_estimation, define_xu_exposure_group
from ..common.hac import estimate_hac_se
from ..common.models import XuEstimateResult
from .propensity_scores import estimate_xu_propensity_scores
from .ipw_weights import compute_xu_ipw_weights
from .ipw_influence import compute_xu_ipw_influence_function
from typing import List, Optional


def compute_xu_ipw_influence_function_ode(
    df: pd.DataFrame,
    neighbors_list: list,
    exposure_type: str,
    config: Config,
    dgp_config: Config,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Calculate ODE influence function for Xu IPW estimator (similar interface to proposed methods)

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

    # 3. Calculate influence function for each g level
    influence_funcs = {}
    for g_val in g_classes:
        # Get propensity scores corresponding to exposure level g
        eta_1g = eta_g.get(1, {}).get(g_val, config.epsilon)
        eta_0g = eta_g.get(0, {}).get(g_val, config.epsilon)

        # Use common IPW weight calculation function
        weights = compute_xu_ipw_weights(
            D=D.values,
            G=G.values,
            eta=eta,
            eta_1g=eta_1g,
            eta_0g=eta_0g,
            g_level=g_val,
        )

        # DATT estimator: normalize by number of treated units
        N_treated = D.sum()
        if N_treated > 0:
            datt_g = np.sum(weights * delta_Y) / N_treated
        else:
            datt_g = 0.0

        # Calculate influence function
        influence_func = compute_xu_ipw_influence_function(
            D=D.values,
            G=G.values,
            delta_Y=delta_Y,
            eta=eta,
            eta_1g=eta_1g,
            eta_0g=eta_0g,
            g_level=g_val,
            X=X,
            ps_d_model=ps_d_model,
            ps_g_model=ps_g_model,
            n=n,
            datt_g=datt_g,
        )
        influence_funcs[g_val] = influence_func

    # 4. Calculate ODE influence function (weighted average)
    p_g_d1 = G[D == 1].value_counts(normalize=True)
    influence_ode = np.zeros(n)
    for g in g_classes:
        if g in influence_funcs:
            influence_ode += influence_funcs[g] * p_g_d1.get(g, 0)

    return influence_ode


def estimate_xu_ipw(
    df: pd.DataFrame,
    neighbors_list: list,
    exposure_type: str,
    config: Config,
    dgp_config: Config,
    locations: np.ndarray,
    K: float,
    random_seed: Optional[int] = None,
) -> XuEstimateResult:
    """Calculate Xu (2025) IPW estimator (for DATT)

    Estimates direct effect DATT(g) (Direct Average Treatment Effect on the Treated)
    for each exposure level and overall direct effect ODE (Overall Direct Effect).

    DATT estimator is normalized by number of treated units to estimate effect on treatment group.
    Weight calculation: treatment group=1, control group=odds ratio.

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        exposure_type: Exposure type ("cs", "mo", "fm")
        config: Configuration object
        dgp_config: DGP configuration object
        locations: Spatial location array
        K: Bandwidth parameter
        random_seed: Optional random seed for reproducible Xu (MO) exposure mapping

    Returns:
        DATT(g) estimates, ODE, standard errors

    References:
        Xu, R. (2025). "Difference-in-Differences with Interference"
    """
    df_est, delta_Y = prepare_data_for_estimation(df)
    # Use only z since z_u is unobserved variable
    X, D = df_est[["z"]], df_est["D"]
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

    # 3. Calculate IPW estimators (DATT(g) for each g)
    # Calculation based on Xu paper equation (8): τ̂^{ipw}(g) = (1/N) Σ_{i=1}^N [(D_i - p̂(Z_i))/(p̂(Z_i)(1-p̂(Z_i))) × I{G_i=g}/(D_iπ̂_{1g}(Z_i)+(1-D_i)π̂_{0g}(Z_i)) × (Y_{i2}-Y_{i1})]
    ipw_ests = {}
    ipw_se = {}
    influence_funcs = {}

    for g_val in g_classes:
        # Get propensity scores corresponding to exposure level g
        eta_1g = eta_g.get(1, {}).get(g_val, config.epsilon)
        eta_0g = eta_g.get(0, {}).get(g_val, config.epsilon)

        # Use common IPW weight calculation function
        weights = compute_xu_ipw_weights(
            D=D.values,
            G=G.values,
            eta=eta,  # eta is already a numpy array from clip_ps()
            eta_1g=eta_1g,
            eta_0g=eta_0g,
            g_level=g_val,
        )

        # DATT estimator: normalize by number of treated units
        N_treated = D.sum()
        if N_treated > 0:
            datt_g = np.sum(weights * delta_Y) / N_treated
        else:
            datt_g = 0.0
        ipw_ests[f"datt_g{g_val}"] = datt_g

        # Calculate influence function
        influence_func = compute_xu_ipw_influence_function(
            D=D.values,
            G=G.values,
            delta_Y=delta_Y,
            eta=eta,
            eta_1g=eta_1g,
            eta_0g=eta_0g,
            g_level=g_val,
            X=X,
            ps_d_model=ps_d_model,
            ps_g_model=ps_g_model,
            n=n,
            datt_g=datt_g,
        )
        influence_funcs[g_val] = influence_func

        # Calculate HAC standard error
        hac_se = estimate_hac_se(influence_func, locations, K, config=config)
        ipw_se[f"datt_g{g_val}_se"] = hac_se

    # 4. Calculate Overall Direct Effect (ODE)
    # Weighted average of DATT(g) by distribution of G in treatment group
    p_g_d1 = G[D == 1].value_counts(normalize=True)
    ode = sum(ipw_ests.get(f"datt_g{g}") * p_g_d1.get(g, 0) for g in g_classes)

    # Calculate ODE influence function (weighted average)
    influence_ode = np.zeros(n)
    for g in g_classes:
        if g in influence_funcs:
            influence_ode += influence_funcs[g] * p_g_d1.get(g, 0)

    # Calculate HAC standard error
    ode_se = estimate_hac_se(influence_ode, locations, K, config=config)

    return XuEstimateResult(
        datt_estimates=ipw_ests,
        ode=ode,
        datt_standard_errors=ipw_se,
        ode_standard_error=ode_se,
    )
