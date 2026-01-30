import numpy as np
import pandas as pd
import warnings
from scipy.linalg import cholesky, LinAlgError
from scipy.special import expit
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict, Optional

# Changed to relative import
from ..utils import find_neighbors
from .config import Config


def _generate_covariates(
    locations: np.ndarray, dgp_config: Config, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Common function to generate covariates z and z_u"""
    n_units = len(locations)

    # Individual attributes z_i ~ i.i.d. N(0, z_std^2)
    z = rng.normal(0, dgp_config.z_std, size=n_units)

    # Generate spatially correlated covariates z_u,i
    # Covariance matrix Sigma (Sigma_ij = correlation_base^{l_A(i,j)})
    dist_matrix = cdist(locations, locations, metric="chebyshev")
    Sigma = np.power(dgp_config.z_u_correlation_base, dist_matrix)

    # Sample from multivariate normal distribution using Cholesky decomposition
    try:
        # Add small value to diagonal for stability (Jittering)
        L = cholesky(Sigma + np.eye(n_units) * 1e-9, lower=True)
        z_u = L @ rng.normal(0, 1, size=n_units)
    except LinAlgError:
        # Fallback to no correlation if matrix is not positive definite (rare)
        z_u = rng.normal(0, 1, size=n_units)

    return z, z_u


def _generate_treatment(
    z: np.ndarray, z_u: np.ndarray, dgp_config: Config, rng: np.random.Generator
) -> np.ndarray:
    """Common function to generate treatment variable D

    Performs probabilistic assignment based on logit model.
    Raises ValueError if balance conditions are not met,
    allowing rejection sampling in the calling function.
    """
    n_units = len(z)

    # Assignment probability p(z*) = logit^-1(treatment_z_coef*z + treatment_z_u_coef*z_u)
    linear_predictor = (
        dgp_config.treatment_z_coef * z + dgp_config.treatment_z_u_coef * z_u
    )
    prob_d = expit(linear_predictor)

    # Probabilistic assignment based on logit model
    # Prioritize theoretical consistency, do not clip treatment probabilities
    # Balance is ensured through rejection sampling (`_validate_generated_data`)
    D = (rng.uniform(0, 1, size=n_units) < prob_d).astype(int)

    # Check balance between treatment and control groups
    treatment_ratio = np.mean(D)
    n_treated = np.sum(D)
    n_control = n_units - n_treated

    # Raise exception if balance conditions are not met
    if not (0.15 <= treatment_ratio <= 0.85 and n_treated >= 10 and n_control >= 10):
        raise ValueError(
            f"Insufficient treatment assignment balance: "
            f"Treatment rate={treatment_ratio:.3f}, Treatment group={n_treated}, Control group={n_control}"
        )

    return D


def _validate_generated_data(D: np.ndarray, S: np.ndarray, dgp_config: Config) -> None:
    """Function to check validity of generated data

    Raises ValueError for serious issues (insufficient treatment/control groups,
    insufficient exposure level diversity), allowing rejection sampling in the calling function.
    """
    # 1. Check balance between treatment and control groups
    treatment_ratio = np.mean(D)
    n_treated = np.sum(D)
    n_control = len(D) - n_treated

    if treatment_ratio < 0.1 or treatment_ratio > 0.9:
        raise ValueError(
            f"Treatment group proportion is extreme: {treatment_ratio:.3f}"
        )

    if n_treated < 5 or n_control < 5:
        raise ValueError(
            f"Too few units in treatment or control group: Treatment group={n_treated}, Control group={n_control}"
        )

    # 2. Check exposure level diversity
    unique_S = np.unique(S)
    max_level = len(dgp_config.spillover_effects) - 1

    # Limit S range to max_level (clip excessively high levels)
    S_clipped = np.clip(S, 0, max_level)

    for level in range(max_level + 1):
        count = np.sum(S_clipped == level)
        if count < 4:  # At least 4 units required per level
            raise ValueError(f"Too few units at exposure level {level}: {count}")

    # 3. Check exposure level distribution in treatment and control groups
    for d_val in [0, 1]:
        mask = D == d_val
        if not mask.any():
            continue

        g_subset = S_clipped[mask]
        unique_g = np.unique(g_subset)
        if len(unique_g) < 2:
            raise ValueError(f"Only one exposure level in D={d_val} group: {unique_g}")

    # 4. Check that both treatment and control groups exist at each exposure level
    for level in range(max_level + 1):
        level_mask = S_clipped == level
        if np.sum(level_mask) > 0:
            level_treated = np.sum(D[level_mask])
            level_control = np.sum(level_mask) - level_treated
            if level_treated < 2 or level_control < 2:
                raise ValueError(
                    f"Insufficient treatment or control group at exposure level {level}: "
                    f"Treatment group={level_treated}, Control group={level_control}"
                )


def _calculate_spillover_effects(S: np.ndarray, dgp_config: Config) -> np.ndarray:
    """Common function to calculate spillover effects (supports dynamic number of levels)"""
    _S = np.clip(S, 0, len(dgp_config.spillover_effects) - 1)
    Spillover = np.array(dgp_config.spillover_effects)[_S]
    return Spillover


def _generate_outcomes(
    z: np.ndarray,
    z_u: np.ndarray,
    D: np.ndarray,
    Spillover: np.ndarray,
    dgp_config: Config,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Common function to generate outcomes Y1 and Y2"""
    n_units = len(z)

    # Y_1i = β_1*z_i + β_2*z_u,i + ε_1i
    Y1 = (
        dgp_config.beta_1 * z
        + dgp_config.beta_2 * z_u
        + rng.normal(0, dgp_config.y1_error_std, size=n_units)
    )

    # Y_2i = δ + Y_1i + τ*D_i + γ_1*z_u,i + γ_2*z_i + f(S_i) + ε_2i
    # ε_2i ~ N(0, 1)
    e2 = rng.normal(0, dgp_config.y2_error_std, size=n_units)

    Y2 = (
        dgp_config.delta
        + Y1
        + dgp_config.tau * D
        + dgp_config.gamma_1 * z_u
        + dgp_config.gamma_2 * z
        + Spillover
        + e2
    )

    return Y1, Y2


def _create_dataframe(
    Y1: np.ndarray,
    Y2: np.ndarray,
    D: np.ndarray,
    z: np.ndarray,
    z_u: np.ndarray,
    S: np.ndarray,
    locations: np.ndarray,
) -> pd.DataFrame:
    """Common function to create dataframe"""
    return pd.DataFrame(
        {
            "Y1": Y1,
            "Y2": Y2,
            "D": D,  # Treatment variable
            "z": z,
            # z_u is unobserved variable, so not included in dataframe
            "S": S,  # Number of treated neighbors
            "x": locations[:, 0],  # Coordinate information
            "y": locations[:, 1],
        }
    )


def _generate_core_data(
    n_units: int,
    K: float,
    space_size: float,
    dgp_config: Config,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, List[List[int]], Dict[str, float]]:
    """Core data generation logic (commonized function)"""
    max_attempts = 10  # Number of retries for data generation

    for attempt in range(max_attempts):
        try:
            # 1. Unit placement and neighborhood definition
            locations = rng.uniform(0, space_size, size=(n_units, 2))
            neighbors_list = find_neighbors(locations, K)

            # 2. Generate covariates
            z, z_u = _generate_covariates(locations, dgp_config, rng)

            # 3. Treatment assignment
            D = _generate_treatment(z, z_u, dgp_config, rng)

            # 4. Calculate number of treated neighbors
            S = np.array([D[n].sum() for n in neighbors_list])

            # 5. Calculate spillover effects
            Spillover = _calculate_spillover_effects(S, dgp_config)

            # 6. Generate outcomes
            Y1, Y2 = _generate_outcomes(z, z_u, D, Spillover, dgp_config, rng)

            # 7. Check data validity
            _validate_generated_data(D, S, dgp_config)

            # 8. Create dataframe
            df = _create_dataframe(Y1, Y2, D, z, z_u, S, locations)

            # 9. Calculate conditional true parameters (changed to pass S)
            true_params = calculate_conditional_params(D, S, neighbors_list, dgp_config)

            return df, neighbors_list, true_params

        except (ValueError, Exception) as e:
            if attempt == max_attempts - 1:
                # Re-raise exception if failed even on final attempt
                raise ValueError(
                    f"Data generation failed (after {max_attempts} attempts): {str(e)}"
                )
            else:
                # Retry
                print(f"Data generation attempt {attempt + 1} failed: {str(e)}")
                continue

    # Should not reach this line, but just in case
    raise ValueError("Data generation failed")


def generate_data(
    n_units: int = 500,
    K: float = 1.0,
    dgp_config: Optional[Config] = None,
    space_size: float = 20.0,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[List[int]], Dict[str, float]]:
    """
    Generate data for simulation (DGP)

    Based on Section 5.1 of the paper and Section 3 of README, generates data
    with spatial interference including spillover effects. Each unit is placed
    in 2D space, and interference occurs between units within neighbor distance K.

    Args:
        n_units: Number of units (N). Default: 400
        K: Neighbor distance (Chebyshev distance). Default: 1.0
        dgp_config: DGP configuration. Uses default settings if None
        space_size: Space size (unit placement range). Default: 20.0
        random_seed: Random seed. Default: None

    Returns:
        Tuple[pd.DataFrame, List[List[int]], Dict[str, float]]:
            - df: Generated dataframe. Contains the following columns:
                - 'x', 'y': Unit coordinates
                - 'z': Confounding factor 1 (N(0,1))
                - 'D': Treatment variable (0 or 1)
                - 'Y1': Outcome at time point 1
                - 'Y2': Outcome at time point 2
            - neighbors_list: List of neighbor indices for each unit
            - true_params: Dictionary of conditional true parameter values

    Raises:
        ImportError: If import of required modules fails
        ValueError: If parameters are invalid

    Example:
        >>> df, neighbors, params = generate_data(n_units=100, K=1.0)
        >>> print(f"Generated {len(df)} units")
        >>> print(f"True ADTT: {params['true_adtt']:.4f}")
    """
    # Initialize DGP configuration
    if dgp_config is None:
        dgp_config = Config()

    # Create random number generator (RNG)
    rng = np.random.default_rng(random_seed)

    # Core logic for data generation
    df, neighbors_list, true_params = _generate_core_data(
        n_units, K, space_size, dgp_config, rng
    )
    return df, neighbors_list, true_params


def calculate_conditional_params(
    D: np.ndarray, S: np.ndarray, neighbors_list: List[List[int]], dgp_config: Config
) -> Dict[str, float]:
    """
    Calculate conditional true ADTT and AITT based on generated data.

    Calculates the "true effect" conditioned on specific data generated in each simulation iteration
    (network structure and covariate realizations) through counterfactual simulation.

    Args:
        D: Treatment variable array
        S: Neighbor treatment count for each unit
        neighbors_list: List of neighbor indices for each unit
        dgp_config: DGP configuration

    Returns:
        Dict[str, float]: Conditional true parameter values
            - "true_adtt": Conditional true ADTT value
            - "true_aitt": Conditional true AITT value
    """
    # ADTT: Equal to direct effect τ (constant)
    true_adtt = dgp_config.tau

    # AITT: Calculate counterfactual effect difference (difference in spillover effects)
    # Definition 2: AITT = (1/n) Σ_i (1/|N_A(i;K)|) Σ_{j∈N_A(i;K)} E[Y_{2j}(D_j, D_{N_A(j;K)}^{(1)}) - Y_{2j}(D_j, D_{N_A(j;K)}^{(0)}) | D_i = 1]
    all_average_neighbor_effects = []
    treatment_units = np.where(D == 1)[0]

    for i in treatment_units:
        neighbors = neighbors_list[i]
        if len(neighbors) == 0:
            continue

        # Convert neighbors list to numpy array
        neighbors_arr = np.array(neighbors)

        # Get actual neighbor treatment count (S_j) for i's neighbors j directly from S
        S_j_actual = S[neighbors_arr]

        # Counterfactual S_j if i was not treated
        # (Assumption: if i is j's neighbor, then j is also i's neighbor (symmetry))
        S_j_counterfactual = S_j_actual - 1

        # Calculate spillover effects at once
        spillover_with_i = _calculate_spillover_effects(S_j_actual, dgp_config)
        spillover_without_i = _calculate_spillover_effects(
            S_j_counterfactual, dgp_config
        )

        # Calculate effect difference
        effects = spillover_with_i - spillover_without_i
        all_average_neighbor_effects.append(np.mean(effects))

    if all_average_neighbor_effects:
        true_aitt = np.mean(all_average_neighbor_effects)
    else:
        true_aitt = 0.0

    # Specify key names (used in evaluation)
    return {"true_adtt": true_adtt, "true_aitt": true_aitt}
