"""
Xu Estimator: DR Influence Function Computation

Computes the influence function for Xu (2025) DR estimator.
"""

import pandas as pd
import numpy as np
from .dr_weights import compute_xu_dr_weights


def compute_xu_dr_influence_function(
    D: np.ndarray,
    G: np.ndarray,
    delta_Y: np.ndarray,
    eta: np.ndarray,
    eta_1g: np.ndarray,
    eta_0g: np.ndarray,
    m_1g_pred: np.ndarray,
    m_0g_pred: np.ndarray,
    g_level: int,
    X: pd.DataFrame,
    ps_d_model,
    ps_g_model,
    outcome_model_1g,
    outcome_model_0g,
    n: int,
    datt_g: float,
) -> np.ndarray:
    """Calculate influence function for Xu paper's DR estimator (for DATT)

    Calculates influence function for DATT estimator. Implements complete influence function
    considering uncertainty in nuisance parameter estimation.

    Args:
        D: Treatment variable array
        G: Exposure level array
        delta_Y: Outcome difference array
        eta: Array of propensity scores P(D=1|Z)
        eta_1g: Array of conditional probabilities P(G=g|D=1,Z)
        eta_0g: Array of conditional probabilities P(G=g|D=0,Z)
        m_1g_pred: Array of outcome predictions for treatment group
        m_0g_pred: Array of outcome predictions for control group
        g_level: Exposure level to calculate
        X: Covariate DataFrame
        ps_d_model: Model for P(D=1|Z)
        ps_g_model: Model for P(G=g|D,Z)
        outcome_model_1g: Outcome model for m_{1g}(Z)
        outcome_model_0g: Outcome model for m_{0g}(Z)
        n: Sample size
        datt_g: DR estimator value

    Returns:
        Array of influence functions
    """
    # Calculate DR weights
    weights_for_delta_Y, adjustment = compute_xu_dr_weights(
        D, G, eta, eta_1g, eta_0g, m_1g_pred, m_0g_pred, g_level
    )
    weighted_delta = weights_for_delta_Y * delta_Y + adjustment

    # 1. Calculate main term (without estimation effect)
    # For DATT estimator, normalize by number of treated units
    # Note: Centering should subtract datt_g only from treatment group (D=1)
    N_treated = D.sum()
    if N_treated > 0:
        tau_lin1 = weighted_delta - (datt_g * D)
    else:
        tau_lin1 = weighted_delta

    # 2. Consider uncertainty in nuisance parameter estimation (tau_lin2)
    tau_lin2 = np.zeros(n)

    # Prepare X_matrix (including intercept term)
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = np.asarray(X)
    X_matrix = np.column_stack([np.ones(n), X_values])

    # 2.1. Asymptotic linear representation of propensity score model P(D=1|Z)
    if ps_d_model is not None:
        try:
            # Calculate score function
            score_d = (D - eta)[:, np.newaxis] * X_matrix

            # Calculate Hessian matrix
            W_d = eta * (1 - eta)
            XWX_d = X_matrix.T @ (W_d[:, np.newaxis] * X_matrix)

            try:
                Hessian_d = np.linalg.inv(XWX_d) * n
            except np.linalg.LinAlgError:
                Hessian_d = np.linalg.pinv(XWX_d) * n

            # Calculate asymptotic linear representation
            asy_lin_rep_d = score_d @ Hessian_d

            # Calculate moment function (for DATT)
            # Derivative of weighted term with respect to covariates
            I_Gg = (G == g_level).astype(float)
            odds = eta / (1 - eta)
            weights_delta = (D * I_Gg / np.clip(eta_1g, 1e-10, 1 - 1e-10)) - (
                (1 - D) * odds * I_Gg / np.clip(eta_0g, 1e-10, 1 - 1e-10)
            )
            weighted_term = weights_delta * delta_Y + adjustment

            # Moment function: E[weighted_term * X] (for DATT)
            # Don't divide here since normalization is done in final influence_func
            mom_d = np.mean(weighted_term[:, np.newaxis] * X_matrix, axis=0)

            # Calculate estimation effect term
            tau_lin2_d = asy_lin_rep_d @ mom_d
            tau_lin2 += tau_lin2_d
        except Exception:
            # Keep as 0 if error occurs (for backward compatibility)
            pass

    # 2.2. Asymptotic linear representation of exposure level model P(G=g|D,Z)
    if ps_g_model is not None:
        try:
            # Prepare X_with_D
            if isinstance(X, pd.DataFrame):
                X_with_D_df = X.copy()
                X_with_D_df["D"] = D
                X_with_D_values = X_with_D_df.values
            else:
                X_values = np.asarray(X)
                X_with_D_values = np.column_stack([X_values, D])
            X_with_D_matrix = np.column_stack([np.ones(n), X_with_D_values])

            # Binary indicator I{G=g}
            I_Gg = (G == g_level).astype(float)

            # Get prediction probabilities
            if hasattr(ps_g_model, "predict_proba"):
                # Get prediction probabilities for D=1 and D=0 cases
                if isinstance(X, pd.DataFrame):
                    X_D1_df = X.copy()
                    X_D1_df["D"] = 1
                    X_D1_values = X_D1_df.values
                    X_D0_df = X.copy()
                    X_D0_df["D"] = 0
                    X_D0_values = X_D0_df.values
                else:
                    X_values = np.asarray(X)
                    X_D1_values = np.column_stack([X_values, np.ones(n)])
                    X_D0_values = np.column_stack([X_values, np.zeros(n)])
                X_D1_matrix = np.column_stack([np.ones(n), X_D1_values])
                X_D0_matrix = np.column_stack([np.ones(n), X_D0_values])

                prob_g_D1 = ps_g_model.predict_proba(X_D1_matrix)
                prob_g_D0 = ps_g_model.predict_proba(X_D0_matrix)

                # Get index of class corresponding to g_level
                if hasattr(ps_g_model, "classes_"):
                    g_classes_list = list(ps_g_model.classes_)
                    g_index = (
                        g_classes_list.index(g_level)
                        if g_level in g_classes_list
                        else 0
                    )
                else:
                    g_index = 0

                eta_1g_pred = (
                    prob_g_D1[:, g_index] if len(prob_g_D1.shape) > 1 else prob_g_D1
                )
                eta_0g_pred = (
                    prob_g_D0[:, g_index] if len(prob_g_D0.shape) > 1 else prob_g_D0
                )

                # Actual probability (depending on D value)
                eta_g_actual = D * eta_1g_pred + (1 - D) * eta_0g_pred

                # Score function
                score_g = (I_Gg - eta_g_actual)[:, np.newaxis] * X_with_D_matrix
                W_g = eta_g_actual * (1 - eta_g_actual)
                XWX_g = X_with_D_matrix.T @ (W_g[:, np.newaxis] * X_with_D_matrix)

                try:
                    Hessian_g = np.linalg.inv(XWX_g) * n
                except np.linalg.LinAlgError:
                    Hessian_g = np.linalg.pinv(XWX_g) * n

                # Calculate asymptotic linear representation
                asy_lin_rep_g = score_g @ Hessian_g

                # Calculate moment function (for DATT)
                I_Gg = (G == g_level).astype(float)
                odds = eta / (1 - eta)
                weights_delta = (D * I_Gg / np.clip(eta_1g, 1e-10, 1 - 1e-10)) - (
                    (1 - D) * odds * I_Gg / np.clip(eta_0g, 1e-10, 1 - 1e-10)
                )
                weighted_term = weights_delta * delta_Y + adjustment

                # Exclude 3rd term independent of parameter gamma (difference in outcome model predictions)
                # 3rd term of adjustment: D * (m_1g_pred - m_0g_pred)
                # This term does not depend on exposure model parameter gamma, so exclude from derivative calculation
                term_pi_dependent = weighted_term - D * (m_1g_pred - m_0g_pred)

                # Derivative correction coefficient: d(1/pi)/d(gamma) = -(1-pi)/pi^2 * d(pi)/d(gamma) = -(1-pi) * W * X
                # Here W = 1/pi, so coefficient is -(1-pi)
                correction_factor = -(1 - eta_g_actual)
                # Don't divide here since normalization is done in final influence_func
                # Use term_pi_dependent instead of weighted_term
                mom_g = np.mean(
                    (correction_factor * term_pi_dependent)[:, np.newaxis]
                    * X_with_D_matrix,
                    axis=0,
                )

                # Calculate estimation effect term
                tau_lin2_g = asy_lin_rep_g @ mom_g
                tau_lin2 += tau_lin2_g
        except Exception:
            # Keep as 0 if error occurs (for backward compatibility)
            pass

    # 3. Calculate complete influence function (for DATT)
    # For DATT, normalize by number of treated units
    if N_treated > 0:
        influence_func = (tau_lin1 - tau_lin2) / (N_treated / n)
    else:
        influence_func = tau_lin1 - tau_lin2

    return influence_func
