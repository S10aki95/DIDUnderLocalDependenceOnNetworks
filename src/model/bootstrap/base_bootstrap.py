"""
Base Bootstrap Class

This module provides a base class for bootstrap implementations to reduce
code duplication between ClusterBootstrap and StandardBootstrap.
Also includes ModelCache for caching propensity scores and outcome models.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from ...settings import Config


class ModelCache:
    """Cache class: uses cluster ID and data hash as keys

    Stores propensity score and outcome model estimation results, and reuses them
    when the same cluster's data is used again.
    """

    def __init__(self):
        """Initialize cache"""
        self._cache: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        self._hit_count = 0
        self._miss_count = 0

    def _compute_hash(self, cluster_data: pd.DataFrame) -> str:
        """Calculate hash of cluster data

        Args:
            cluster_data: Cluster dataframe

        Returns:
            Data hash value (hexadecimal string)
        """
        # Sort dataframe and hash (independent of row order)
        # However, column order is preserved
        sorted_data = cluster_data.sort_values(by=cluster_data.columns.tolist())

        # Convert numeric data to string and hash
        # Use pandas hash_pandas_object for more efficient method
        try:
            # Available in pandas 1.4.0+
            hash_values = pd.util.hash_pandas_object(sorted_data, index=False)
            hash_str = str(hash_values.sum())
        except AttributeError:
            # Fallback for older pandas versions
            data_str = sorted_data.to_string()
            hash_str = data_str

        # Calculate MD5 hash
        return hashlib.md5(hash_str.encode()).hexdigest()

    def get_cache_key(
        self, cluster_id: Any, cluster_data: pd.DataFrame
    ) -> Tuple[Any, str]:
        """Generate cache key

        Args:
            cluster_id: Cluster ID
            cluster_data: Cluster dataframe

        Returns:
            Tuple of (cluster_id, data_hash)
        """
        data_hash = self._compute_hash(cluster_data)
        return (cluster_id, data_hash)

    def get(
        self, cluster_id: Any, cluster_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Get from cache

        Args:
            cluster_id: Cluster ID
            cluster_data: Cluster dataframe

        Returns:
            Dictionary of cached results (ps_results, outcome_predictions),
            or None if not in cache
        """
        cache_key = self.get_cache_key(cluster_id, cluster_data)

        if cache_key in self._cache:
            self._hit_count += 1
            return self._cache[cache_key]
        else:
            self._miss_count += 1
            return None

    def set(
        self,
        cluster_id: Any,
        cluster_data: pd.DataFrame,
        ps_results: Dict[str, np.ndarray],
        outcome_predictions: Dict[str, np.ndarray],
    ) -> None:
        """Save to cache

        Args:
            cluster_id: Cluster ID
            cluster_data: Cluster dataframe
            ps_results: Propensity score estimation results
            outcome_predictions: Outcome model estimation results
        """
        cache_key = self.get_cache_key(cluster_id, cluster_data)
        self._cache[cache_key] = {
            "ps_results": ps_results,
            "outcome_predictions": outcome_predictions,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Cache statistics (hit count, miss count, cache size)
        """
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "cache_size": len(self._cache),
            "hit_rate": (
                self._hit_count / (self._hit_count + self._miss_count)
                if (self._hit_count + self._miss_count) > 0
                else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear cache"""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0


class BaseBootstrap:
    """Base class for bootstrap

    Provides common parallel execution, progress bar, and standard error calculation logic.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize BaseBootstrap

        Args:
            config: Config object (uses default Config() if None)
        """
        if config is None:
            from ...settings import Config

            config = Config()
        self.config = config

    def _suppress_stdout(self):
        """Suppress standard output (prevent log output during parallel execution)"""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def _restore_stdout(self):
        """Restore standard output"""
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def _compute_standard_errors(
        self,
        bootstrap_estimates: List[Dict[str, float]],
        se_suffix: str = "_se",
    ) -> Dict[str, float]:
        """Calculate standard errors from bootstrap estimation results

        Args:
            bootstrap_estimates: List of bootstrap estimation results
            se_suffix: Suffix to add to standard error keys (default: "_se")

        Returns:
            Dictionary of standard errors
        """
        if not bootstrap_estimates:
            print("Error: No valid bootstrap samples")
            return {}

        # Calculate standard errors
        bootstrap_df = pd.DataFrame(bootstrap_estimates)

        standard_errors = {}
        skipped_columns = []

        for col in bootstrap_df.columns:
            # Process only numeric data
            if bootstrap_df[col].dtype in ["int64", "float64", "int32", "float32"]:
                # Warn if NaN included
                if bootstrap_df[col].isna().any():
                    nan_count = bootstrap_df[col].isna().sum()
                    print(
                        f"Warning: Column '{col}' contains {nan_count} NaN values. Calculating standard error excluding NaN."
                    )
                    # Calculate standard error excluding NaN
                    valid_values = bootstrap_df[col].dropna()
                    if len(valid_values) > 0:
                        standard_errors[f"{col}{se_suffix}"] = valid_values.std()
                    else:
                        print(
                            f"Warning: Column '{col}' has no valid values. Skipping standard error calculation."
                        )
                        skipped_columns.append(col)
                else:
                    standard_errors[f"{col}{se_suffix}"] = bootstrap_df[col].std()
            else:
                # Skip dictionary types and other non-numeric data
                print(
                    f"Warning: Column '{col}' is not numeric type ({bootstrap_df[col].dtype}), skipping standard error calculation."
                )
                skipped_columns.append(col)

        return standard_errors

    def _generate_iteration_seeds(
        self,
        base_seed: Optional[int] = None,
        n_bootstrap: int = 100,
    ) -> np.ndarray:
        """Generate independent seeds for each iteration

        Args:
            base_seed: Base random seed (get from config if None)
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Array of seeds for each iteration
        """
        if base_seed is None:
            base_seed = getattr(self.config, "random_seed", 42)
        rng_seed = np.random.default_rng(base_seed)
        return rng_seed.integers(0, 2**31, size=n_bootstrap)

    def _run_parallel_bootstrap(
        self,
        n_bootstrap: int,
        iteration_func: Callable,
        iteration_args: List[tuple],
        n_jobs: Optional[int] = None,
        progress_desc: str = "Running bootstrap",
        progress_bar: Optional[Any] = None,
    ) -> List[Any]:
        """Execute parallel bootstrap (with progress bar)

        Args:
            n_bootstrap: Number of bootstrap iterations
            iteration_func: Function to execute each iteration
            iteration_args: List of arguments to pass to each iteration
            n_jobs: Number of jobs for parallel execution (use CPU core count if None)
            progress_desc: Progress bar description
            progress_bar: External progress bar (optional, updates progress if specified)

        Returns:
            List of bootstrap estimation results
        """
        # Determine number of jobs for parallel execution
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() // 2)

        # Update description if progress bar specified (don't display message)
        # Only update if progress bar is not disabled
        original_desc = None
        if progress_bar and not getattr(progress_bar, "disable", False):
            original_desc = progress_bar.desc
            # Update progress bar description
            progress_bar.set_description(f"{original_desc} - Running bootstrap")

        # Parallel execution (without progress bar)
        # Call iteration_func directly (don't pass unpicklable objects)
        # Use backend='threading' to avoid pickling errors on Windows
        # Threading backend doesn't require pickling, so large objects are fine
        bootstrap_estimates = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(iteration_func)(*args) for args in iteration_args
        )

        # Restore description if progress bar specified and not disabled
        if (
            progress_bar
            and original_desc
            and not getattr(progress_bar, "disable", False)
        ):
            progress_bar.set_description(original_desc)

        # Exclude None (errors)
        bootstrap_estimates = [est for est in bootstrap_estimates if est is not None]

        return bootstrap_estimates
