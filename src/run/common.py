"""
Common estimator calculation module

Provides common estimator calculation logic used in both simulation and real data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from ..settings import Config
from ..model.standard import estimate_ipw, estimate_dr_did
from ..model.proposed import (
    compute_adtt_influence_function,
    compute_aitt_influence_function,
    compute_dr_adtt_influence_function,
    compute_dr_aitt_influence_function,
    estimate_proposed_with_se,
)
from ..model.xu import estimate_xu_dr, estimate_xu_ipw

# Constant definitions
ESTIMATOR_COLUMNS = {
    "Proposed IPW ADTT (Logistic)": "proposed_adtt",
    "Proposed IPW AITT (Logistic)": "proposed_aitt",
    "Proposed DR ADTT (Logistic)": "proposed_dr_adtt",
    "Proposed DR AITT (Logistic)": "proposed_dr_aitt",
    "Canonical IPW": "canonical_ipw",
    "Canonical TWFE": "canonical_twfe",
    "DR-DID": "dr_did",
    "Modified TWFE": "modified_twfe",
    "Xu DR (CS) - ODE": "xu_dr_cs_ode",
    "Xu DR (MO) - ODE": "xu_dr_mo_ode",
    "Xu DR (FM) - ODE": "xu_dr_fm_ode",
    "Xu IPW (CS) - ODE": "xu_ipw_cs_ode",
    "Xu IPW (MO) - ODE": "xu_ipw_mo_ode",
    "Xu IPW (FM) - ODE": "xu_ipw_fm_ode",
}

SE_COLUMNS = {
    "Proposed IPW ADTT (Logistic)": "proposed_adtt_se",
    "Proposed IPW AITT (Logistic)": "proposed_aitt_se",
    "Proposed DR ADTT (Logistic)": "proposed_dr_adtt_se",
    "Proposed DR AITT (Logistic)": "proposed_dr_aitt_se",
    "Canonical IPW": "canonical_ipw_se",
    "Canonical TWFE": "canonical_twfe_se",
    "DR-DID": "dr_did_se",
    "Modified TWFE": "modified_twfe_se",
    "Xu DR (CS) - ODE": "xu_dr_cs_ode_se",
    "Xu DR (MO) - ODE": "xu_dr_mo_ode_se",
    "Xu DR (FM) - ODE": "xu_dr_fm_ode_se",
    "Xu IPW (CS) - ODE": "xu_ipw_cs_ode_se",
    "Xu IPW (MO) - ODE": "xu_ipw_mo_ode_se",
    "Xu IPW (FM) - ODE": "xu_ipw_fm_ode_se",
}


def _generate_iteration_seeds(base_seed: int, n_simulations: int) -> np.ndarray:
    """
    Helper function to generate independent seeds for each iteration

    Args:
        base_seed: Base random seed
        n_simulations: Number of simulations

    Returns:
        Array of seeds for each iteration
    """
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 2**32 - 1, size=n_simulations)


def _print_simulation_config(config: Config):
    """
    Helper function to display simulation configuration

    Args:
        config: Configuration object
    """
    print(f"\nSimulation configuration:")
    print(f"  Number of simulations: {config.n_simulations}")
    print(f"  Number of units: {config.n_units}")
    print(f"  Neighbor distance (K): {config.k_distance}")
    print(f"  Space size: {config.space_size}")
    print(f"  Random seed: {config.random_seed}")


def _generate_experiment_id(
    overrides: Optional[Dict[str, Any]], experiment_id: Optional[str] = None
) -> str:
    """
    Helper function to generate experiment ID

    Args:
        overrides: Dictionary of override settings
        experiment_id: Explicitly specified experiment ID

    Returns:
        Experiment ID string (prefixed with underscore)
    """
    if experiment_id is not None:
        return f"_{experiment_id}"

    if overrides:
        param_name = list(overrides.keys())[0]
        param_value = overrides[param_name]
        return f"_{param_name}_{str(param_value).replace('.', '_')}"

    return ""


def compute_all_estimators(
    df: pd.DataFrame,
    neighbors_list: List[List[int]],
    config: Config,
    locations: Optional[np.ndarray] = None,
    K: Optional[float] = None,
    covariates: Optional[List[str]] = None,
    treatment_col: str = "D",
    compute_standard_se: bool = True,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Common function to calculate all estimators

    Provides a unified interface usable in both simulation and real data analysis.

    Args:
        df: Dataframe (required columns: Y1, Y2, treatment_col, covariates)
        neighbors_list: Neighbor list
        config: Configuration object
        locations: Unit coordinates (for HAC standard error, optional)
        K: Neighbor distance (for HAC standard error, optional)
        covariates: List of covariate column names (default: ["z"])
        treatment_col: Treatment variable column name (default: "D")
        compute_standard_se: Whether to calculate standard error (default: True)
        random_seed: Optional random seed for reproducible Xu (MO) exposure mapping

    Returns:
        Dictionary of estimator results
    """
    if covariates is None:
        # Determine default covariates
        excluded_cols = {"Y1", "Y2", treatment_col, "x", "y", "G"}
        if "z" in df.columns:
            covariates = ["z"]
        else:
            covariates = [col for col in df.columns if col not in excluded_cols]

    results = {}

    # 1. Proposed methods (ADTT/AITT)
    try:
        adtt_influence = compute_adtt_influence_function(
            df,
            neighbors_list,
            "logistic",
            config,
            covariates=covariates,
            treatment_col=treatment_col,
            random_seed=random_seed,
        )
        results["proposed_adtt"] = np.mean(adtt_influence)

        aitt_influence = compute_aitt_influence_function(
            df,
            neighbors_list,
            "logistic",
            config,
            covariates=covariates,
            treatment_col=treatment_col,
            random_seed=random_seed,
        )
        results["proposed_aitt"] = np.mean(aitt_influence)

        # Calculate standard error
        if compute_standard_se:
            if locations is not None and K is not None:
                # HAC standard error (when coordinate information is available)
                adtt_result = estimate_proposed_with_se(
                    df,
                    neighbors_list,
                    locations,
                    K,
                    compute_adtt_influence_function,
                    "logistic",
                    config,
                    random_seed=random_seed,
                )
                aitt_result = estimate_proposed_with_se(
                    df,
                    neighbors_list,
                    locations,
                    K,
                    compute_aitt_influence_function,
                    "logistic",
                    config,
                    random_seed=random_seed,
                )
                results["proposed_adtt_se"] = adtt_result.standard_error
                results["proposed_aitt_se"] = aitt_result.standard_error
            else:
                # Calculate standard error from standard deviation of influence function (for real data)
                n = len(adtt_influence)
                results["proposed_adtt_se"] = np.std(adtt_influence, ddof=1) / np.sqrt(
                    n
                )
                results["proposed_aitt_se"] = np.std(aitt_influence, ddof=1) / np.sqrt(
                    n
                )

        # Calculate DR estimators
        if locations is not None and K is not None:
            try:
                dr_adtt_result = estimate_proposed_with_se(
                    df,
                    neighbors_list,
                    locations,
                    K,
                    compute_dr_adtt_influence_function,
                    "logistic",
                    config,
                    random_seed=random_seed,
                )
                results["proposed_dr_adtt"] = dr_adtt_result.estimate
                if compute_standard_se:
                    results["proposed_dr_adtt_se"] = dr_adtt_result.standard_error

                dr_aitt_result = estimate_proposed_with_se(
                    df,
                    neighbors_list,
                    locations,
                    K,
                    compute_dr_aitt_influence_function,
                    "logistic",
                    config,
                    random_seed=random_seed,
                )
                results["proposed_dr_aitt"] = dr_aitt_result.estimate
                if compute_standard_se:
                    results["proposed_dr_aitt_se"] = dr_aitt_result.standard_error
            except Exception as e:
                if config.verbose:
                    print(f"Warning: Proposed DR estimator calculation failed: {e}")
    except Exception as e:
        if config.verbose:
            print(f"Warning: Proposed method calculation failed: {e}")

    # 2. Standard methods (Canonical IPW, DR-DID)
    try:
        ipw_result = estimate_ipw(
            df, config, covariates=covariates, treatment_col=treatment_col
        )
        results["canonical_ipw"] = ipw_result.estimate
        if compute_standard_se:
            results["canonical_ipw_se"] = ipw_result.standard_error
    except Exception as e:
        if config.verbose:
            print(f"Warning: Canonical IPW calculation failed: {e}")

    try:
        dr_did_result = estimate_dr_did(
            df, config, covariates=covariates, treatment_col=treatment_col
        )
        results["dr_did"] = dr_did_result.estimate
        if compute_standard_se:
            results["dr_did_se"] = dr_did_result.standard_error
    except Exception as e:
        if config.verbose:
            print(f"Warning: DR-DID calculation failed: {e}")

    # 3. Xu estimators (only when coordinate information is available)
    if locations is not None and K is not None:
        for exposure_type in ["cs", "mo", "fm"]:
            try:
                xu_dr_result = estimate_xu_dr(
                    df,
                    neighbors_list,
                    exposure_type,
                    config,
                    config,
                    locations,
                    K,
                    random_seed=random_seed,
                )
                results[f"xu_dr_{exposure_type}_ode"] = xu_dr_result.ode
                if compute_standard_se:
                    results[f"xu_dr_{exposure_type}_ode_se"] = (
                        xu_dr_result.ode_standard_error
                    )
            except Exception as e:
                if config.verbose:
                    print(f"Warning: Xu DR ({exposure_type}) calculation failed: {e}")

            try:
                xu_ipw_result = estimate_xu_ipw(
                    df,
                    neighbors_list,
                    exposure_type,
                    config,
                    config,
                    locations,
                    K,
                    random_seed=random_seed,
                )
                results[f"xu_ipw_{exposure_type}_ode"] = xu_ipw_result.ode
                if compute_standard_se:
                    results[f"xu_ipw_{exposure_type}_ode_se"] = (
                        xu_ipw_result.ode_standard_error
                    )
            except Exception as e:
                if config.verbose:
                    print(f"Warning: Xu IPW ({exposure_type}) calculation failed: {e}")

    return results
