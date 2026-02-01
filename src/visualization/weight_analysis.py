"""
Influence function analysis module

Provides functions to collect influence functions for each method.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from ..settings import Config


def collect_influence_functions(
    df: pd.DataFrame,
    neighbors_list: List[list],
    config: Config,
    dgp_config: Config,
    exposure_type: str = "cs",
    covariates: List[str] = None,
    treatment_col: str = "D",
    random_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Collect influence functions for each method

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        config: Configuration object
        dgp_config: DGP configuration object
        exposure_type: Exposure type for Xu method ("cs", "mo", "fm")
        covariates: List of covariate column names (default: ["z"])
        treatment_col: Treatment variable column name (default: "D")
        random_seed: Optional random seed for reproducible Xu (MO) exposure mapping.
                     Defaults to config.random_seed if None.

    Returns:
        Dictionary containing influence function arrays for each method
    """
    # Use config.random_seed as default if random_seed not provided
    if random_seed is None:
        random_seed = getattr(config, "random_seed", None)
    if covariates is None:
        covariates = ["z"]

    influence_funcs = {}

    # 1. Influence function for proposed method ADTT
    from src.model.proposed.ipw_adtt_influence import compute_adtt_influence_function

    adtt_influence = compute_adtt_influence_function(
        df,
        neighbors_list,
        "logistic",
        config,
        covariates=covariates,
        treatment_col=treatment_col,
        random_seed=random_seed,
    )
    influence_funcs["Proposed IPW (ADTT)"] = adtt_influence

    # 2. Influence function for proposed method AITT
    from src.model.proposed.ipw_aitt_influence import compute_aitt_influence_function

    aitt_influence = compute_aitt_influence_function(
        df,
        neighbors_list,
        "logistic",
        config,
        covariates=covariates,
        treatment_col=treatment_col,
        random_seed=random_seed,
    )
    influence_funcs["Proposed IPW (AITT)"] = aitt_influence

    # 2-1. Influence function for proposed method DR ADTT
    from src.model.proposed.dr_adtt_influence import compute_dr_adtt_influence_function

    dr_adtt_influence = compute_dr_adtt_influence_function(
        df,
        neighbors_list,
        "logistic",
        config,
        covariates=covariates,
        treatment_col=treatment_col,
        random_seed=random_seed,
    )
    influence_funcs["Proposed DR (ADTT)"] = dr_adtt_influence

    # 2-2. Influence function for proposed method DR AITT
    from src.model.proposed.dr_aitt_influence import compute_dr_aitt_influence_function

    dr_aitt_influence = compute_dr_aitt_influence_function(
        df,
        neighbors_list,
        "logistic",
        config,
        covariates=covariates,
        treatment_col=treatment_col,
        random_seed=random_seed,
    )
    influence_funcs["Proposed DR (AITT)"] = dr_aitt_influence

    # 3. Xu IPW influence function (for ODE)
    from src.model.xu.ipw_estimator import compute_xu_ipw_influence_function_ode

    xu_ipw_influence = compute_xu_ipw_influence_function_ode(
        df,
        neighbors_list,
        exposure_type,
        config,
        dgp_config,
        covariates=covariates,
        treatment_col=treatment_col,
        random_seed=random_seed,
    )
    influence_funcs[f"Xu IPW ({exposure_type.upper()})"] = xu_ipw_influence

    # 4. Xu DR influence function (for ODE)
    from src.model.xu.dr_estimator import compute_xu_dr_influence_function_ode

    xu_dr_influence = compute_xu_dr_influence_function_ode(
        df,
        neighbors_list,
        exposure_type,
        config,
        dgp_config,
        covariates=covariates,
        treatment_col=treatment_col,
        random_seed=random_seed,
    )
    influence_funcs[f"Xu DR ({exposure_type.upper()})"] = xu_dr_influence

    # 5. Xu DR influence function (no adjustment version, for ODE)
    from src.model.xu.dr_estimator import (
        compute_xu_dr_influence_function_ode_no_adjustment,
    )

    xu_dr_influence_no_adj = compute_xu_dr_influence_function_ode_no_adjustment(
        df,
        neighbors_list,
        exposure_type,
        config,
        dgp_config,
        covariates=covariates,
        treatment_col=treatment_col,
        random_seed=random_seed,
    )
    influence_funcs[f"Xu DR ({exposure_type.upper()}) - No Adjustment"] = (
        xu_dr_influence_no_adj
    )

    return influence_funcs


def collect_influence_functions_from_simulation(
    results_list: List[Dict],
    df_list: List[pd.DataFrame],
    neighbors_list_list: List[List[list]],
    config: Config,
    dgp_config: Config,
    exposure_type: str = "cs",
) -> pd.DataFrame:
    """Collect influence functions from simulation results

    Args:
        results_list: List of simulation results
        df_list: List of dataframes
        neighbors_list_list: List of neighbor lists
        config: Configuration object
        dgp_config: DGP configuration object
        exposure_type: Exposure type for Xu method ("cs", "mo", "fm")

    Returns:
        DataFrame containing influence function data
    """
    all_influence_funcs = []

    for i, (df, neighbors_list) in enumerate(zip(df_list, neighbors_list_list)):
        try:
            influence_dict = collect_influence_functions(
                df,
                neighbors_list,
                config,
                dgp_config,
                exposure_type=exposure_type,
            )

            for method, influence_array in influence_dict.items():
                for unit_idx, influence_value in enumerate(influence_array):
                    all_influence_funcs.append(
                        {
                            "simulation": i,
                            "unit": unit_idx,
                            "method": method,
                            "influence_function": influence_value,
                            "abs_influence_function": np.abs(influence_value),
                        }
                    )
        except Exception as e:
            print(
                f"Warning: Failed to collect influence functions for simulation {i}: {e}"
            )
            continue

    return pd.DataFrame(all_influence_funcs)
