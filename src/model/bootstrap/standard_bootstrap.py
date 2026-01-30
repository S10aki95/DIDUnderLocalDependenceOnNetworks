"""
Standard Bootstrap for Standard Error Estimation

This module provides standard bootstrap functionality for simulations where
units are sampled independently (non-clustered bootstrap).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import fields

from ...settings import Config
from ...run.common import compute_all_estimators
from .base_bootstrap import BaseBootstrap


# Top-level function: to make it picklable
def _remap_neighbors_list_static(
    neighbors_list: List[List[int]],
    bootstrap_indices: np.ndarray,
) -> List[List[int]]:
    """Remap neighbor list to match bootstrap sample indices (static version)

    Args:
        neighbors_list: Original neighbor list
        bootstrap_indices: Array of bootstrap sample indices

    Returns:
        Remapped neighbor list
    """
    n_bootstrap = len(bootstrap_indices)
    # Create mapping from original index to new index
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(bootstrap_indices)}

    remapped_neighbors = []
    for i in range(n_bootstrap):
        old_unit_idx = bootstrap_indices[i]
        old_neighbors = neighbors_list[old_unit_idx]
        # Map to new indices (only those included in bootstrap sample)
        new_neighbors = [
            index_map[neighbor_idx]
            for neighbor_idx in old_neighbors
            if neighbor_idx in index_map
        ]
        remapped_neighbors.append(new_neighbors)

    return remapped_neighbors


def _bootstrap_iteration_static(
    df: pd.DataFrame,
    neighbors_list: List[List[int]],
    config_dict: dict,
    locations: Optional[np.ndarray],
    K: Optional[float],
    covariates: Optional[List[str]],
    treatment_col: str,
    iteration_seed: int,
) -> Optional[Dict[str, float]]:
    """Execute one bootstrap iteration (static function for parallelization)

    Args:
        df: Original dataframe
        neighbors_list: Original neighbor list
        config_dict: Dictionary representation of Config object (to make it picklable)
        locations: Unit coordinates (for HAC standard error, optional)
        K: Neighbor distance (for HAC standard error, optional)
        covariates: List of covariate column names
        treatment_col: Treatment variable column name
        iteration_seed: Seed for this iteration

    Returns:
        Dictionary of estimation results, or None on error
    """
    # Reconstruct Config object (avoid pickling issues)
    from ...settings import Config

    config = Config(**config_dict)

    # Don't suppress standard output with threading backend (shared between threads)
    # Instead, set verbose=False in compute_all_estimators to suppress log output

    try:
        # Set random state with independent seed
        rng = np.random.default_rng(iteration_seed)

        # Sample each unit independently (standard bootstrap)
        n_units = len(df)
        bootstrap_indices = rng.choice(n_units, size=n_units, replace=True)

        # Create bootstrap sample
        bootstrap_df = df.iloc[bootstrap_indices].reset_index(drop=True)

        # Remap neighbor list
        remapped_neighbors_list = _remap_neighbors_list_static(
            neighbors_list, bootstrap_indices
        )

        # Also remap coordinates (if they exist)
        bootstrap_locations = None
        if locations is not None:
            bootstrap_locations = locations[bootstrap_indices]

        # Calculate estimators (suppress log output with verbose=False)
        # Temporarily set config.verbose to False
        original_verbose = config.verbose
        config.verbose = False

        try:
            results = compute_all_estimators(
                df=bootstrap_df,
                neighbors_list=remapped_neighbors_list,
                config=config,
                locations=bootstrap_locations,
                K=K,
                covariates=covariates,
                treatment_col=treatment_col,
                compute_standard_se=False,  # Don't calculate standard error within bootstrap
                random_seed=iteration_seed,  # Pass iteration seed for reproducible Xu (MO) exposure mapping
            )
        finally:
            # Restore verbose
            config.verbose = original_verbose

        return results
    except Exception as e:
        # Return None on error (handled by caller)
        return None


class StandardBootstrap(BaseBootstrap):
    """Standard bootstrap class

    Implements normal bootstrap where each unit is sampled independently.
    Used for standard error estimation in simulations.
    """

    def _remap_neighbors_list(
        self,
        neighbors_list: List[List[int]],
        bootstrap_indices: np.ndarray,
    ) -> List[List[int]]:
        """Remap neighbor list to match bootstrap sample indices

        Args:
            neighbors_list: Original neighbor list
            bootstrap_indices: Array of bootstrap sample indices

        Returns:
            Remapped neighbor list
        """
        n_bootstrap = len(bootstrap_indices)
        # Create mapping from original index to new index
        index_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(bootstrap_indices)
        }

        remapped_neighbors = []
        for i in range(n_bootstrap):
            old_unit_idx = bootstrap_indices[i]
            old_neighbors = neighbors_list[old_unit_idx]
            # Map to new indices (only those included in bootstrap sample)
            new_neighbors = [
                index_map[neighbor_idx]
                for neighbor_idx in old_neighbors
                if neighbor_idx in index_map
            ]
            remapped_neighbors.append(new_neighbors)

        return remapped_neighbors

    def _bootstrap_iteration(
        self,
        df: pd.DataFrame,
        neighbors_list: List[List[int]],
        config: Config,
        locations: Optional[np.ndarray],
        K: Optional[float],
        covariates: Optional[List[str]],
        treatment_col: str,
        iteration_seed: int,
    ) -> Optional[Dict[str, float]]:
        """Execute one bootstrap iteration (internal function for parallelization)

        Args:
            df: Original dataframe
            neighbors_list: Original neighbor list
            config: Config object
            locations: Unit coordinates (for HAC standard error, optional)
            K: Neighbor distance (for HAC standard error, optional)
            covariates: List of covariate column names
            treatment_col: Treatment variable column name
            iteration_seed: Seed for this iteration

        Returns:
            Dictionary of estimation results, or None on error
        """
        # Suppress standard output (prevent log output during parallel execution)
        self._suppress_stdout()

        try:
            # Set random state with independent seed
            rng = np.random.default_rng(iteration_seed)

            # Sample each unit independently (standard bootstrap)
            n_units = len(df)
            bootstrap_indices = rng.choice(n_units, size=n_units, replace=True)

            # Create bootstrap sample
            bootstrap_df = df.iloc[bootstrap_indices].reset_index(drop=True)

            # Remap neighbor list
            remapped_neighbors_list = self._remap_neighbors_list(
                neighbors_list, bootstrap_indices
            )

            # Also remap coordinates (if they exist)
            bootstrap_locations = None
            if locations is not None:
                bootstrap_locations = locations[bootstrap_indices]

            # Calculate estimators
            results = compute_all_estimators(
                df=bootstrap_df,
                neighbors_list=remapped_neighbors_list,
                config=config,
                locations=bootstrap_locations,
                K=K,
                covariates=covariates,
                treatment_col=treatment_col,
                compute_standard_se=False,  # Don't calculate standard error within bootstrap
                random_seed=iteration_seed,  # Pass iteration seed for reproducible Xu (MO) exposure mapping
            )

            return results
        except Exception as e:
            # Return None on error (handled by caller)
            return None
        finally:
            # Restore standard output
            self._restore_stdout()

    def bootstrap_standard_errors(
        self,
        df: pd.DataFrame,
        neighbors_list: List[List[int]],
        config: Config,
        locations: Optional[np.ndarray] = None,
        K: Optional[float] = None,
        covariates: Optional[List[str]] = None,
        treatment_col: str = "D",
        n_jobs: Optional[int] = None,
        progress_bar: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Estimate standard errors using standard bootstrap method (with parallelization support)

        Args:
            df: DataFrame
            neighbors_list: Neighbors list
            config: Config object
            locations: Unit coordinates (for HAC standard errors, optional)
            K: Neighbor distance (for HAC standard errors, optional)
            covariates: List of covariate column names
            treatment_col: Treatment variable column name
            n_jobs: Number of jobs for parallel execution (if None, use CPU core count)

        Returns:
            Dictionary of standard errors (keys in `*_se_bootstrap` format)
        """
        # Get bootstrap count from config
        n_bootstrap = config.n_bootstrap

        # Generate seeds
        iteration_seeds = self._generate_iteration_seeds(
            base_seed=getattr(config, "random_seed", 42),
            n_bootstrap=n_bootstrap,
        )

        # Prepare arguments for parallel execution (convert Config to dictionary to make it pickleable)
        # Use fields() to extract only fields (exclude attributes added in __post_init__)
        config_dict = {
            field.name: getattr(config, field.name) for field in fields(config)
        }
        iteration_args = [
            (
                df,
                neighbors_list,
                config_dict,  # Convert Config to dictionary
                locations,
                K,
                covariates,
                treatment_col,
                iteration_seeds[b],
            )
            for b in range(n_bootstrap)
        ]

        # Parallel execution (use static function directly)
        # Pass progress bar to display progress (do not display messages)
        bootstrap_estimates = self._run_parallel_bootstrap(
            n_bootstrap=n_bootstrap,
            iteration_func=_bootstrap_iteration_static,
            iteration_args=iteration_args,
            n_jobs=n_jobs,
            progress_desc="Running bootstrap",
            progress_bar=progress_bar,
        )

        if not bootstrap_estimates:
            return {}

        # Calculate standard errors (use base class method, specify suffix)
        return self._compute_standard_errors(
            bootstrap_estimates, se_suffix="_se_bootstrap"
        )
