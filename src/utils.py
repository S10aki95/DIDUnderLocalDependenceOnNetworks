"""
Module providing general-purpose utility functions

This module provides general-purpose functionality independent of estimators:
- Spatial neighbor search
- Data preprocessing
- Machine learning model retrieval
- Evaluation metric calculation
- Basic numerical processing

Estimator-specific logic is located in estimators.py.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from typing import List, Dict, Optional, Tuple, Any, Literal, Callable, Union
from sklearn.linear_model import LogisticRegression, LinearRegression
from .settings import Config


def find_neighbors(locations: np.ndarray, K: float) -> List[List[int]]:
    """
    Identify neighbor list based on distances between units

    Uses Chebyshev distance (L∞ norm) to identify neighbors for each unit.
    Defines units with distance ≤ K as neighbors, excluding self.

    Args:
        locations: Unit coordinate array (N x d). N is number of units, d is dimensionality
        K: Distance threshold for defining neighbors (Chebyshev distance)

    Returns:
        List[List[int]]: List of neighbor indices for each unit.
        Each element is a list of indices of neighbor units for that unit

    Raises:
        ValueError: If locations is empty array

    Example:
        >>> locations = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> neighbors = find_neighbors(locations, K=1.0)
        >>> print(neighbors[0])  # [1, 2] (neighbors of unit 0)
    """
    # Calculate distance matrix (Chebyshev distance L-infinity norm)
    # Corresponds to definition in paper Section 5.1: rho(i,j) = max{|s_1i-s_1j|, |s_2i-s_2j|}
    dist_matrix = cdist(locations, locations, metric="chebyshev")

    n_units = locations.shape[0]
    neighbors_list = []

    for i in range(n_units):
        # Distance ≤ K, exclude self (distance 0)
        # For floating point comparison, ensure strictly greater than 0 (1e-9 is a small value)
        neighbor_mask = (dist_matrix[i] <= K) & (dist_matrix[i] > 1e-9)
        neighbor_indices = np.where(neighbor_mask)[0]

        # Sort by distance (following specification in README Section 3.3.1)
        if len(neighbor_indices) > 0:
            neighbor_distances = dist_matrix[i][neighbor_indices]
            sorted_indices = neighbor_indices[np.argsort(neighbor_distances)]
            neighbors_list.append(sorted_indices.tolist())
        else:
            neighbors_list.append([])

    return neighbors_list


# =============================================================================
# Data preprocessing
# =============================================================================


def clip_ps(ps: np.ndarray, config: Config) -> np.ndarray:
    """Clip propensity scores"""
    return np.clip(ps, config.ps_clip_min, config.ps_clip_max)


def prepare_data_for_estimation(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Preprocess data for estimation"""
    df_est = df.copy()
    df_est["delta_Y"] = df_est["Y2"] - df_est["Y1"]
    delta_Y = df_est["delta_Y"].values
    return df_est, delta_Y


# =============================================================================
# Machine learning models
# =============================================================================


def get_ml_model(
    model_type: Literal["logistic", "linear"],
    config: Config,
):
    """Return machine learning model instance"""
    if model_type == "logistic":
        return LogisticRegression(
            solver="saga",  # Solver suitable for large-scale data
            max_iter=config.logistic_max_iter,
            multi_class="auto",
        )
    if model_type == "linear":
        return LinearRegression()


def estimate_unified_outcome_model(
    D: np.ndarray,
    G: np.ndarray,
    X: np.ndarray,
    delta_Y: np.ndarray,
    config: Config,
) -> Dict[str, np.ndarray]:
    """Unified outcome model estimation function

    Implements approach from real_data.py: train single model on all data,
    using features including D, G, X, and D*X interaction terms.

    Args:
        D: Treatment variable array (n,)
        G: Exposure level array (n,)
        X: Covariate array (n, p)
        delta_Y: Outcome difference array (n,)
        config: Config object

    Returns:
        Dictionary of predictions. Keys are in format `m_delta_{d}{g}` (d, g are 0 or 1)
    """
    n_covariates = X.shape[1]

    # Create feature matrix (D, G, X, D*X interaction terms)
    interaction_terms = np.column_stack([D * X[:, j] for j in range(n_covariates)])
    X_features = np.column_stack([D, G, X, interaction_terms])

    # Train single model on all data
    model_delta = get_ml_model("linear", config)
    model_delta.fit(X_features, delta_Y)

    # Generate predictions for each (D,G) combination
    predictions = {}
    for d in [0, 1]:
        for g in [0, 1]:
            # Create interaction terms
            interaction_pred = np.column_stack(
                [np.full(len(X), d) * X[:, j] for j in range(n_covariates)]
            )
            X_pred = np.column_stack(
                [
                    np.full(len(X), d),
                    np.full(len(X), g),
                    X,
                    interaction_pred,
                ]
            )
            delta_pred = model_delta.predict(X_pred)
            predictions[f"m_delta_{d}{g}"] = delta_pred

    return predictions


def estimate_outcome_model_from_features(
    D: np.ndarray,
    X_features_df: pd.DataFrame,
    delta_Y: np.ndarray,
    config: Config,
) -> Dict[str, np.ndarray]:
    """Outcome model estimation function based on neighbor treatment vector

    Based on paper definition (Section 3.2, Eq. 306-307),
    estimates Outcome Model conditioning on entire neighbor treatment vector D_{N_A(i;K)}.

    μ_{1i}(D_{N_A(i;K)}, z_i) = E[Y_{2i} - Y_{1i} | D_i=1, D_{N_A(i;K)}, z_i]
    μ_{0i}(D_{N_A(i;K)}, z_i) = E[Y_{2i} - Y_{1i} | D_i=0, D_{N_A(i;K)}, z_i]

    Args:
        D: Treatment variable array (n,)
        X_features_df: Feature DataFrame containing neighbor treatment vectors (n, p)
                       Usually created by create_neighbor_features
        delta_Y: Outcome difference array (n,)
        config: Config object

    Returns:
        Dictionary of predictions. Keys are `m_delta_0` and `m_delta_1`
        - `m_delta_0`: Predictions for D_i=0 case (n,)
        - `m_delta_1`: Predictions for D_i=1 case (n,)
    """
    n = len(D)
    X_features = X_features_df.values

    # To generate predictions for both D=0 and D=1 cases,
    # add D to features and train model
    # Features: format [D, X_features]
    X_with_D = np.column_stack([D, X_features])

    # Train single model on all data
    model_delta = get_ml_model("linear", config)
    model_delta.fit(X_with_D, delta_Y)

    # Predictions for D=0 case
    X_pred_D0 = np.column_stack([np.zeros(n), X_features])
    mu_0_pred = model_delta.predict(X_pred_D0)

    # Predictions for D=1 case
    X_pred_D1 = np.column_stack([np.ones(n), X_features])
    mu_1_pred = model_delta.predict(X_pred_D1)

    return {
        "m_delta_0": mu_0_pred,
        "m_delta_1": mu_1_pred,
    }


# =============================================================================
# Evaluation metric calculation
# =============================================================================


def calculate_bias(
    estimates: np.ndarray, true_values: Union[float, np.ndarray]
) -> float:
    """Calculate Bias. Supports cases where true values vary (conditional)."""
    return np.mean(estimates - true_values)


def calculate_rmse(
    estimates: np.ndarray, true_values: Union[float, np.ndarray]
) -> float:
    """Calculate RMSE (Root Mean Squared Error). Supports cases where true values vary (conditional)."""
    return np.sqrt(np.mean((estimates - true_values) ** 2))


def calculate_coverage_rate(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    true_values: Union[float, np.ndarray],
    z_score: Optional[float] = None,
    confidence_level: Optional[float] = None,
) -> float:
    """Calculate Coverage Rate. Supports cases where true values vary (conditional).

    Args:
        estimates: Array of estimates
        standard_errors: Array of standard errors
        true_values: True values (scalar or array)
        z_score: Z-value (use this if specified)
        confidence_level: Confidence level (used when z_score is not specified, default: 0.95)
    """
    # Calculate from confidence_level if z_score not specified
    if z_score is None:
        if confidence_level is None:
            confidence_level = 0.95
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)

    ci_lower = estimates - z_score * standard_errors
    ci_upper = estimates + z_score * standard_errors
    # Check if true value is contained for each iteration
    coverage = np.mean((true_values >= ci_lower) & (true_values <= ci_upper))
    return coverage


def evaluate_estimators(
    results_df: pd.DataFrame,
    estimator_columns: Dict[str, str],
    se_columns: Optional[Dict[str, str]] = None,
    config: Optional["Config"] = None,
) -> pd.DataFrame:
    """Batch calculate evaluation metrics for multiple estimators (supports varying true values)

    Args:
        results_df: Dataframe of simulation results
        estimator_columns: Mapping of estimator names to column names
        se_columns: Mapping of standard error column names (optional)
        config: Configuration object (optional, to get z_critical)
    """
    evaluation_results = []

    # Determine z_score (get from config or use default value)
    if config is not None:
        z_score = config.z_critical
    else:
        # Calculate from default confidence level if config not provided
        z_score = norm.ppf(1 - (1 - 0.95) / 2)

    for estimator_name, column_name in estimator_columns.items():
        if column_name not in results_df.columns:
            continue

        estimates = results_df[column_name].values
        valid_mask = ~np.isnan(estimates)
        if not valid_mask.any():
            continue

        valid_estimates = estimates[valid_mask]

        # Select true value column name (match key names set in dgp.py)
        if "adtt" in estimator_name.lower() or "att" in estimator_name.lower():
            true_value_col = "true_adtt"
        elif "aitt" in estimator_name.lower():
            true_value_col = "true_aitt"
        else:
            true_value_col = "true_adtt"

        if true_value_col not in results_df.columns:
            continue

        # Get true values for each iteration from results_df
        true_values = results_df[true_value_col].values[valid_mask]

        # Calculate evaluation metrics (pass arrays)
        bias = calculate_bias(valid_estimates, true_values)
        rmse = calculate_rmse(valid_estimates, true_values)

        result = {
            "Estimator": estimator_name,
            "Bias": bias,
            "RMSE": rmse,
            "N_Valid": len(valid_estimates),
        }

        # Also calculate coverage rate if standard errors available
        if se_columns and estimator_name in se_columns:
            se_column = se_columns[estimator_name]
            if se_column in results_df.columns:
                valid_se = results_df[se_column].values[valid_mask]
                coverage = calculate_coverage_rate(
                    valid_estimates, valid_se, true_values, z_score=z_score
                )
                result["Coverage_Rate"] = coverage

        evaluation_results.append(result)

    return pd.DataFrame(evaluation_results)


# =============================================================================
# Feature creation
# =============================================================================


def define_xu_exposure_group(
    df: pd.DataFrame,
    neighbors_list: List[list],
    exposure_type: str,
    dgp_config: Config,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """Define exposure group (G) for Xu estimators

    Args:
        df: Dataframe
        neighbors_list: Neighbor list
        exposure_type: Exposure type ("cs", "mo", "fm")
        dgp_config: DGP configuration object
        rng: Optional random number generator for reproducible random assignment (MO only).
             If None, uses global np.random for backward compatibility.

    Returns:
        Exposure group series
    """
    # Calculate number of treated units in neighbors for each unit
    s_i = np.array([df["D"].iloc[n].sum() for n in neighbors_list])

    # Get spillover effects from DGP config passed as argument
    spillover_effects = dgp_config.spillover_effects

    # Maximum level index of true structure (e.g., 3 if 4 levels)
    true_max_level = len(spillover_effects) - 1

    if exposure_type in ["CS", "cs"]:  # Correctly Specified
        # Fix: Correctly identify based on true structure (0-3 if 4 levels)
        return pd.Series(np.clip(s_i, 0, true_max_level), index=df.index)

    if exposure_type in ["MO", "mo"]:  # Misspecified Ordering
        # Fix: Clip with true structure
        g_mo = pd.Series(np.clip(s_i, 0, true_max_level), index=df.index)

        # Randomly assign 30% of units to different groups
        n_units = len(df)
        n_random = int(0.3 * n_units)

        # Use provided RNG if available, otherwise use global np.random for backward compatibility
        if rng is not None:
            random_indices = rng.choice(n_units, size=n_random, replace=False)
            g_mo.iloc[random_indices] = rng.choice(
                range(true_max_level + 1), size=n_random
            )
        else:
            random_indices = np.random.choice(n_units, n_random, replace=False)
            g_mo.iloc[random_indices] = np.random.choice(
                range(true_max_level + 1), n_random
            )

        return pd.Series(g_mo, index=df.index)

    if exposure_type in ["FM", "fm"]:  # Fully Misspecified
        # Simplify to 2 levels (present/absent)
        return pd.Series((s_i > 1).astype(int), index=df.index)

    raise ValueError(f"Invalid exposure_type: {exposure_type}")


def create_neighbor_features(
    df: pd.DataFrame,
    neighbors_list: List[list],
    config: Config,
    for_aitt: bool = False,
    pairs: Optional[List[tuple]] = None,
    covariates: List[str] = ["z"],
    treatment_col: str = "D",
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Create feature matrix for proposed methods (common for ADTT, AITT)

    Theorem 1 (ADTT): e_i = Pr(D_i=1 | D_{N_A(i;K)}, z_i)
    Theorem 2 (AITT): e'_{ij} = Pr(D_i=1 | D_j, D_{N_A(j;K)}^{-i}, z_i, z_j)

    Args:
        for_aitt: True if for AITT estimation, False for ADTT
        pairs: list of (i, j) tuples for AITT estimation (required when for_aitt=True)
        covariates: List of covariate column names (default: ["z"])
        treatment_col: Treatment variable column name (default: "D")
        rng: Optional random number generator for random neighbor sampling (default: None)
    """
    D = df[treatment_col].values

    # Use covariates
    Z = df[covariates].values
    covariate_names = covariates

    max_feats = config.max_neighbors

    # Check neighbors_list length
    if len(neighbors_list) != len(df):
        raise ValueError(
            f"neighbors_list length ({len(neighbors_list)}) must match df length ({len(df)})"
        )

    if not for_aitt:  # For ADTT: Theorem 1
        # e_i = Pr(D_i=1 | D_{N_A(i;K)}, z_i)
        # Features: [z_i, D_{neighbor1}, D_{neighbor2}, ...]
        features = np.zeros((len(df), max_feats))
        for i, neighbors in enumerate(neighbors_list):
            if neighbors:
                # Check neighbor index range
                if any(n < 0 or n >= len(df) for n in neighbors):
                    raise IndexError(
                        f"Invalid neighbor indices for unit {i}: {neighbors}. "
                        f"All indices must be in range [0, {len(df)})"
                    )
                # Randomly sample max_feats neighbors if rng is provided and len > max_feats
                if rng is not None and len(neighbors) > max_feats:
                    selected_indices = rng.choice(
                        len(neighbors), size=max_feats, replace=False
                    )
                    selected_neighbors = [neighbors[idx] for idx in selected_indices]
                    n_treats = D[selected_neighbors]
                else:
                    n_treats = D[neighbors][:max_feats]
                features[i, : len(n_treats)] = n_treats

        # Set feature names and return as DataFrame
        feature_names = covariate_names + [f"neighbor_{i}" for i in range(max_feats)]
        feature_data = np.hstack([Z, features])
        return pd.DataFrame(feature_data, columns=feature_names)
    else:  # For AITT: Theorem 2
        # e'_{ij} = Pr(D_i=1 | D_j, D_{N_A(j;K)}^{-i}, z_i, z_j)
        # Features: [z_i, z_j, D_j, D_{neighbor1(excl. i)}, D_{neighbor2(excl. i)}, ...]
        if pairs is None:
            raise ValueError("pairs must be provided when for_aitt=True")

        if len(pairs) == 0:
            # Return empty DataFrame for empty pair list
            z_i_names = [f"{name}_i" for name in covariate_names]
            z_j_names = [f"{name}_j" for name in covariate_names]
            feature_names = (
                z_i_names
                + z_j_names
                + [treatment_col + "_j"]
                + [f"neighbor_{k}" for k in range(max_feats)]
            )
            return pd.DataFrame(columns=feature_names)

        X_pairs_list = []
        for i, j in pairs:
            # Check index range
            if i < 0 or i >= len(df) or j < 0 or j >= len(df):
                raise IndexError(
                    f"Invalid pair indices: ({i}, {j}). Must be in range [0, {len(df)})"
                )

            neighbors_of_j_minus_i = [n for n in neighbors_list[j] if n != i]
            # Check neighbor index range
            if any(n < 0 or n >= len(df) for n in neighbors_of_j_minus_i):
                raise IndexError(
                    f"Invalid neighbor indices for unit j={j} (excluding i={i}): "
                    f"{neighbors_of_j_minus_i}. All indices must be in range [0, {len(df)})"
                )
            # Randomly sample max_feats neighbors if rng is provided and len > max_feats
            if rng is not None and len(neighbors_of_j_minus_i) > max_feats:
                selected_indices = rng.choice(
                    len(neighbors_of_j_minus_i), size=max_feats, replace=False
                )
                selected_neighbors = [
                    neighbors_of_j_minus_i[idx] for idx in selected_indices
                ]
                D_neighbors = D[selected_neighbors]
            else:
                D_neighbors = D[neighbors_of_j_minus_i][:max_feats]
            padded_D_neighbors = np.pad(
                D_neighbors, (0, max_feats - len(D_neighbors)), mode="constant"
            )

            # Combine covariates (z_i and z_j)
            z_i_names = [f"{name}_i" for name in covariate_names]
            z_j_names = [f"{name}_j" for name in covariate_names]
            feature_data = np.concatenate([Z[i], Z[j], [D[j]], padded_D_neighbors])
            X_pairs_list.append(feature_data)

        # Set feature names and return as DataFrame
        feature_names = (
            z_i_names
            + z_j_names
            + [treatment_col + "_j"]
            + [f"neighbor_{k}" for k in range(max_feats)]
        )
        return pd.DataFrame(X_pairs_list, columns=feature_names)
