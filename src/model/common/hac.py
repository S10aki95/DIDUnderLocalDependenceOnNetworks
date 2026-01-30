"""
HAC (Heteroskedasticity and Autocorrelation Consistent) Standard Error Estimation

This module provides utilities for computing HAC standard errors used across
different estimation methods.
"""

import numpy as np
from typing import Optional, Callable
from scipy.spatial.distance import cdist
from ...settings import Config


def estimate_hac_se(
    influence_func: np.ndarray,
    locations: Optional[np.ndarray] = None,
    K: Optional[float] = None,
    bandwidth: Optional[float] = None,
    config: Optional[Config] = None,
    distance_func: Optional[Callable[[int, int], float]] = None,
    dist_matrix: Optional[np.ndarray] = None,
) -> float:
    """Estimate HAC standard error

    Calculates standard error considering spatial correlation. Based on formula in paper Section 3.2:

    V̂_n = Σ_{s≥0} ω(s/b_n) Ω̂_n(s)

    where ω(·) is Bartlett kernel: ω(x) = 1-|x| for |x| ≤ 1

    Args:
        influence_func: Array of influence functions
        locations: Spatial location array (used when dist_matrix is None)
        K: Bandwidth parameter (used when dist_matrix is None)
        bandwidth: Explicit bandwidth specification (optional)
        config: Configuration object (optional)
        distance_func: Distance calculation function (deprecated: not used. Please construct dist_matrix directly)
        dist_matrix: Distance matrix (N x N). If provided, other parameters are ignored

    Returns:
        HAC standard error
    """
    N = len(influence_func)

    # Determine distance matrix construction method and bandwidth
    if dist_matrix is not None:
        # When distance matrix is directly provided
        if dist_matrix.shape != (N, N):
            raise ValueError(f"dist_matrix must be ({N}, {N}), got {dist_matrix.shape}")
        # Determine bandwidth
        if bandwidth is not None:
            b = bandwidth
        elif config is not None:
            b = config.hac_bandwidth_multiplier * config.hac_default_max_distance
        else:
            raise ValueError(
                "Either bandwidth or config must be provided when using dist_matrix"
            )
    elif distance_func is not None:
        # distance_func parameter is deprecated (implementation removed)
        # Recommend constructing distance matrix directly and passing as dist_matrix parameter
        raise ValueError(
            "distance_func parameter is deprecated. "
            "Please construct the distance matrix directly and pass it as dist_matrix parameter. "
            "See _estimate_hac_se_within_county in real_data.py for an example."
        )
    else:
        # Calculate distance from locations and K
        if locations is None or K is None:
            raise ValueError(
                "Either dist_matrix or both locations and K must be provided"
            )
        # Determine bandwidth
        if bandwidth is not None:
            b = bandwidth
        elif config is not None:
            b = config.hac_bandwidth_multiplier * K
        else:
            raise ValueError(
                "Either bandwidth or config must be provided when using locations and K"
            )
        # Construct distance matrix (using cdist, Chebyshev distance)
        dist_matrix = cdist(locations, locations, metric="chebyshev")

    # Centering: standard procedure to correct bias in variance estimator in finite samples
    Z_centered = influence_func - np.mean(influence_func)

    # Calculate HAC variance
    # Paper formula: V̂_n = (1/N) * Σ_i Σ_j Z_i * Z_j * w(d_ij/b)
    # where w(d/b) = max(0, 1 - d/b) is Bartlett kernel

    # Calculate Bartlett kernel weight matrix: w(d/b) = max(0, 1 - d/b)
    kernel_weights = np.maximum(0, 1 - dist_matrix / b)

    # Calculate HAC variance (vectorized): V = (1/N) * Z' * W * Z
    # Implementation of paper formula: V̂_n = Σ_{s≥0} ω(s/b_n) Ω̂_n(s)
    V_hac = (1 / N) * (Z_centered.T @ kernel_weights @ Z_centered)

    # Standard error = sqrt(V_hac / N)
    return np.sqrt(V_hac / N)
