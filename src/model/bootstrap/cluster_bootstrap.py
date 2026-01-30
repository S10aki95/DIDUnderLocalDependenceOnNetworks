"""
Cluster Bootstrap for Standard Error Estimation

This module provides cluster bootstrap functionality with caching for
propensity scores and outcome models to improve computational efficiency.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from ...settings import Config
from .base_bootstrap import BaseBootstrap, ModelCache


# Top-level function: to make it picklable
def _cluster_bootstrap_iteration_static(
    df: pd.DataFrame,
    config: Config,
    PS_covariates: List[str],
    Z_list: List[str],
    clusters: np.ndarray,
    n_clusters: int,
    cluster_col: str,
    iteration_seed: int,
    cache: Optional[ModelCache] = None,
) -> Optional[Dict[str, float]]:
    """Execute one bootstrap iteration (static function for parallelization)

    Args:
        df: Original dataframe
        config: Config object (used to create new SEZEstimator instance in each worker)
        PS_covariates: List of covariates for propensity score
        Z_list: List of covariates for outcome model
        clusters: Array of clusters (counties)
        n_clusters: Number of clusters
        cluster_col: Cluster column name
        iteration_seed: Seed for this iteration
        cache: Cache object (optional)

    Returns:
        Dictionary of estimation results, or None on error
    """
    # Suppress standard output (prevent log output during parallel execution)
    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
    except Exception:
        pass

    try:
        # Create new SEZEstimator instance in each worker (avoid pickling issues)
        from ...run.real_data import SEZEstimator

        estimator = SEZEstimator(config=config)

        # Set random state with independent seed
        rng = np.random.default_rng(iteration_seed)

        # Sample clusters with replacement
        bootstrap_clusters = rng.choice(clusters, size=n_clusters, replace=True)

        # Create bootstrap sample
        bootstrap_data = []
        for cluster_id in bootstrap_clusters:
            cluster_data = df[df[cluster_col] == cluster_id]
            bootstrap_data.append(cluster_data)
        bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)

        # Process when using cache
        if cache is not None:
            # Check cache for each cluster
            # Get required columns
            required_cols_ps = list(set(PS_covariates + ["W", "G"]))

            # Get list of unique cluster IDs (considering same cluster may be selected multiple times)
            unique_clusters = np.unique(bootstrap_clusters)

            # Check cache for each unique cluster
            for cluster_id in unique_clusters:
                cluster_data_ps = df[df[cluster_col] == cluster_id][
                    required_cols_ps
                ].dropna()

                # Try to get from cache
                cached = cache.get(cluster_id, cluster_data_ps)

                if cached is None:
                    # If not in cache, estimate and save to cache
                    cluster_data_full = df[df[cluster_col] == cluster_id]
                    ps_results = estimator.estimate_propensity_scores(
                        cluster_data_full, PS_covariates
                    )
                    outcome_predictions = estimator.estimate_outcome_models(
                        cluster_data_full, Z_list
                    )

                    # Save to cache
                    cache.set(
                        cluster_id,
                        cluster_data_ps,
                        ps_results,
                        outcome_predictions,
                    )

            # Calculate estimators for entire bootstrap sample
            # Cache is saved per cluster, so normal estimation is needed for entire bootstrap sample
            results = estimator.estimate_all_effects(
                bootstrap_df, PS_covariates, Z_list
            )
        else:
            # When not using cache (existing implementation)
            bootstrap_data = []
            for cluster in bootstrap_clusters:
                cluster_data = df[df[cluster_col] == cluster]
                bootstrap_data.append(cluster_data)

            bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)

            # Estimate effects
            results = estimator.estimate_all_effects(
                bootstrap_df, PS_covariates, Z_list
            )

        return results
    except Exception as e:
        # Return None on error (handled by caller)
        return None
    finally:
        # Restore standard output
        try:
            sys.stdout.close()
            sys.stdout = original_stdout
        except Exception:
            pass


# Top-level function: to make it picklable
def _cluster_iteration_wrapper(
    df_arg,
    config_arg,
    PS_covariates_arg,
    Z_list_arg,
    clusters_arg,
    n_clusters_arg,
    cluster_col_arg,
    iteration_seed_arg,
    cache_arg,
):
    """Wrapper function for cluster bootstrap (picklable)

    Creates new cache instance in each worker.
    """
    # Create new cache instance in each worker
    worker_cache = ModelCache() if cache_arg is not None else None
    return _cluster_bootstrap_iteration_static(
        df_arg,
        config_arg,
        PS_covariates_arg,
        Z_list_arg,
        clusters_arg,
        n_clusters_arg,
        cluster_col_arg,
        iteration_seed_arg,
        cache=worker_cache,
    )


class ClusterBootstrap(BaseBootstrap):
    """Cluster bootstrap class

    Generates bootstrap samples by cluster and estimates standard errors.
    Caches propensity score and outcome model estimation results to improve computational efficiency.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize ClusterBootstrap

        Args:
            config: Config object (uses default Config() if None)
        """
        super().__init__(config)
        self.cache = ModelCache()

    def _bootstrap_iteration(
        self,
        df: pd.DataFrame,
        config: Config,
        PS_covariates: List[str],
        Z_list: List[str],
        clusters: np.ndarray,
        n_clusters: int,
        cluster_col: str,
        iteration_seed: int,
        cache: Optional[ModelCache] = None,
    ) -> Optional[Dict[str, float]]:
        """Execute one bootstrap iteration (internal function for parallelization)

        Args:
            df: Original dataframe
            config: Config object (used to create new SEZEstimator instance in each worker)
            PS_covariates: List of covariates for propensity score
            Z_list: List of covariates for outcome model
            clusters: Array of clusters (counties)
            n_clusters: Number of clusters
            cluster_col: Cluster column name
            iteration_seed: Seed for this iteration
            cache: Cache object (optional)

        Returns:
            Dictionary of estimation results, or None on error
        """
        # Suppress standard output (prevent log output during parallel execution)
        self._suppress_stdout()

        try:
            # Create new SEZEstimator instance in each worker (avoid pickling issues)
            from ...run.real_data import SEZEstimator

            estimator = SEZEstimator(config=config)

            # Set random state with independent seed
            rng = np.random.default_rng(iteration_seed)

            # Sample clusters with replacement
            bootstrap_clusters = rng.choice(clusters, size=n_clusters, replace=True)

            # Create bootstrap sample
            bootstrap_data = []
            for cluster_id in bootstrap_clusters:
                cluster_data = df[df[cluster_col] == cluster_id]
                bootstrap_data.append(cluster_data)
            bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)

            # Process when using cache
            if cache is not None:
                # Check cache for each cluster
                # Get required columns
                required_cols_ps = list(set(PS_covariates + ["W", "G"]))

                # Get list of unique cluster IDs (considering same cluster may be selected multiple times)
                unique_clusters = np.unique(bootstrap_clusters)

                # Check cache for each unique cluster
                for cluster_id in unique_clusters:
                    cluster_data_ps = df[df[cluster_col] == cluster_id][
                        required_cols_ps
                    ].dropna()

                    # Try to get from cache
                    cached = cache.get(cluster_id, cluster_data_ps)

                    if cached is None:
                        # If not in cache, estimate and save to cache
                        cluster_data_full = df[df[cluster_col] == cluster_id]
                        ps_results = estimator.estimate_propensity_scores(
                            cluster_data_full, PS_covariates
                        )
                        outcome_predictions = estimator.estimate_outcome_models(
                            cluster_data_full, Z_list
                        )

                        # Save to cache
                        cache.set(
                            cluster_id,
                            cluster_data_ps,
                            ps_results,
                            outcome_predictions,
                        )

                # Calculate estimators for entire bootstrap sample
                # Cache is saved per cluster, so normal estimation is needed for entire bootstrap sample
                results = estimator.estimate_all_effects(
                    bootstrap_df, PS_covariates, Z_list
                )
            else:
                # When not using cache (existing implementation)
                bootstrap_data = []
                for cluster in bootstrap_clusters:
                    cluster_data = df[df[cluster_col] == cluster]
                    bootstrap_data.append(cluster_data)

                bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)

                # Estimate effects
                results = estimator.estimate_all_effects(
                    bootstrap_df, PS_covariates, Z_list
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
        estimator: Any,  # SEZEstimator type but use Any to avoid circular import
        PS_covariates: List[str],
        Z_list: List[str],
        cluster_col: str = "cty",
        n_jobs: Optional[int] = None,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """Estimate standard errors using cluster bootstrap method (parallelization supported)

        Args:
            df: Dataframe
            estimator: SEZEstimator instance
            PS_covariates: List of covariates for propensity score
            Z_list: List of covariates for outcome model
            cluster_col: Cluster column name (default: "cty")
            n_jobs: Number of jobs for parallel execution (use CPU core count if None)
            use_cache: Whether to use cache (default: True)

        Returns:
            Dictionary of standard errors
        """
        # Get number of bootstrap iterations from config
        n_bootstrap = estimator.config.n_bootstrap

        print(f"Running cluster bootstrap method... (n_bootstrap={n_bootstrap})")
        if use_cache:
            print("Cache feature: enabled")

        # Get list of clusters (counties)
        clusters = df[cluster_col].unique()
        n_clusters = len(clusters)

        # Generate seeds
        iteration_seeds = self._generate_iteration_seeds(
            base_seed=getattr(estimator.config, "random_seed", 42),
            n_bootstrap=n_bootstrap,
        )

        # Share cache (each worker uses independent cache)
        cache_to_use = self.cache if use_cache else None

        # Prepare arguments for parallel execution
        config = estimator.config
        iteration_args = [
            (
                df,
                config,
                PS_covariates,
                Z_list,
                clusters,
                n_clusters,
                cluster_col,
                iteration_seeds[b],
                cache_to_use,
            )
            for b in range(n_bootstrap)
        ]

        # Parallel execution
        bootstrap_estimates = self._run_parallel_bootstrap(
            n_bootstrap=n_bootstrap,
            iteration_func=_cluster_iteration_wrapper,
            iteration_args=iteration_args,
            n_jobs=n_jobs,
            progress_desc="Running bootstrap",
        )

        if not bootstrap_estimates:
            return {}

        # Debug: Check which estimators are included in first iteration
        if bootstrap_estimates:
            first_estimate = bootstrap_estimates[0]
            print(f"\nDebug: Estimators detected in first bootstrap iteration:")
            print(f"  Number of estimators: {len(first_estimate)}")
            print(f"  Estimator keys: {list(first_estimate.keys())}")

        # Debug: Check DataFrame columns and data types
        bootstrap_df = pd.DataFrame(bootstrap_estimates)
        print(f"\nDebug: DataFrame columns and data types:")
        for col in bootstrap_df.columns:
            dtype = bootstrap_df[col].dtype
            has_nan = bootstrap_df[col].isna().any()
            print(f"  {col}: {dtype} (contains NaN: {has_nan})")

        # Calculate standard errors (using base class method)
        standard_errors = self._compute_standard_errors(
            bootstrap_estimates, se_suffix="_se"
        )

        # Display cache statistics
        if use_cache:
            cache_stats = self.cache.get_stats()
            print(f"\nCache statistics:")
            print(f"  Hit count: {cache_stats['hit_count']}")
            print(f"  Miss count: {cache_stats['miss_count']}")
            print(f"  Cache size: {cache_stats['cache_size']}")
            print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")

        return standard_errors
