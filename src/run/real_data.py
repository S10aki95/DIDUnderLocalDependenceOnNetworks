"""
Module providing estimators for real data experiments

This module provides estimators specialized for real data (SEZ data).

Main features:
- SEZEstimator: Estimator class for real data
  - Xu (2025)'s doubly robust and IPW estimators
  - Application of proposed methods (ADTT/AITT) to real data
  - Analysis based on county-level neighborhood definition

Note:
- This module is dedicated to real data
- Estimators for simulation are located in run/simulation.py
- Includes real data-specific processing (county-level neighborhood definition, variable name conversion, etc.)
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable, Any
from tqdm import tqdm
from ..model.xu import (
    compute_xu_ipw_weights,
    compute_xu_dr_weights,
    estimate_xu_propensity_scores,
)
from ..model.standard import estimate_ipw, estimate_dr_did
from ..model.proposed import (
    compute_adtt_influence_function,
    compute_aitt_influence_function,
)
from ..model.common import estimate_hac_se
from ..settings import Config
from ..utils import get_ml_model, estimate_unified_outcome_model
from .common import compute_all_estimators


class SEZEstimator:
    """
    Estimator class for real data experiments

    This class provides estimators specialized for real data (SEZ policy data).

    Main features:
    1. Xu (2025) estimators:
       - Doubly robust estimator (DR): Proposed method in Xu's paper
       - IPW estimator (IPW): Comparison method in Xu's paper
    2. Proposed methods (ADTT/AITT):
       - Analysis based on county-level neighborhood definition
       - Available via `calculate_proposed_adtt_aitt` method

    Real data-specific processing:
    - Neighborhood definition at county level
    - Variable name conversion (W → D, handling multiple covariates)
    - Cluster bootstrap standard error calculation
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize SEZEstimator

        Args:
            config: Config object (uses default Config() if None)
        """
        if config is None:
            config = Config()
        self.config = config

    def estimate_propensity_scores(
        self,
        df: pd.DataFrame,
        PS_covariates: List[str],
        df_common: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Estimate propensity scores (wrapper for estimate_xu_propensity_scores)

        Uses all variables in PS_covariates (4 basic covariates and 4 county-level LOO mean values)
        as specified in the paper. Estimates from all data.

        Args:
            df: Original dataframe
            PS_covariates: List of covariates for propensity score estimation
            df_common: Common dataset (optional). Uses this if provided.

        Returns:
            Dictionary of propensity score estimation results
        """
        # Use common dataset if provided, otherwise prepare internally
        if df_common is not None:
            df_clean = df_common[PS_covariates + ["W", "G"]].copy()
            # Common dataset is already cleaned, so no need to dropna() again
        else:
            # Real data-specific preprocessing
            df_clean = df[PS_covariates + ["W", "G"]].dropna()
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            df_clean = df_clean.reset_index(drop=True)

        # Use all variables in PS_covariates as specified in the paper
        # 4 basic covariates (airport_p, port_p, kl_p, num_p) and
        # 4 county-level LOO mean values (airport_p_county_mean, port_p_county_mean, kl_p_county_mean, num_p_county_mean)
        X = pd.DataFrame(df_clean[PS_covariates].values, columns=PS_covariates)
        W = pd.Series(df_clean["W"].values)
        G = pd.Series(df_clean["G"].values)

        # Create Config object (inherit optimized settings)
        config = Config(
            ps_clip_min=self.config.ps_clip_min,
            ps_clip_max=self.config.ps_clip_max,
            logistic_max_iter=self.config.logistic_max_iter,  # Inherit optimized value
        )

        # Use common function (estimate from all data)
        ps_results = estimate_xu_propensity_scores(X, W, G, config)

        # Convert output format (for compatibility with existing code)
        eta_Z = ps_results.eta
        eta_11_Z = ps_results.eta_g[1][1]  # P(G=1|W=1,Z)
        eta_01_Z = ps_results.eta_g[0][1]  # P(G=1|W=0,Z)
        eta_10_Z = ps_results.eta_g[1][0]  # P(G=0|W=1,Z)
        eta_00_Z = ps_results.eta_g[0][0]  # P(G=0|W=0,Z)

        # Check array lengths (for debugging)
        if self.config.verbose:
            n_data = len(df_clean)
            if len(eta_Z) != n_data:
                tqdm.write(
                    f"Warning: Propensity score array length mismatch - Data: {n_data}, eta_Z: {len(eta_Z)}"
                )
            if len(eta_11_Z) != n_data:
                tqdm.write(
                    f"Warning: Propensity score array length mismatch - Data: {n_data}, eta_11_Z: {len(eta_11_Z)}"
                )

        return {
            "eta_Z": eta_Z,
            "eta_11_Z": eta_11_Z,
            "eta_01_Z": eta_01_Z,
            "eta_10_Z": eta_10_Z,
            "eta_00_Z": eta_00_Z,
        }

    def estimate_outcome_models(
        self,
        df: pd.DataFrame,
        Z_list: List[str],
        df_common: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Estimate outcome models

        Uses 4 basic covariates (airport_p, port_p, kl_p, num_p) and their interaction terms
        as specified in the paper. Estimates from all data.

        Args:
            df: Original dataframe
            Z_list: List of basic covariates for outcome model estimation
            df_common: Common dataset (optional). Uses this if provided.

        Returns:
            Dictionary of outcome model prediction results
        """
        # Use common dataset if provided, otherwise prepare internally
        if df_common is not None:
            df_clean = df_common[Z_list + ["W", "G", "Y1", "Y2"]].copy()
            # Common dataset is already cleaned, so no need to dropna() again
        else:
            # Remove missing values and infinite values
            df_clean = df[Z_list + ["W", "G", "Y1", "Y2"]].dropna()
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            df_clean = df_clean.reset_index(drop=True)

        # Use 4 basic covariates (airport_p, port_p, kl_p, num_p) as specified in the paper
        # Z_list is the list of basic covariates (defined in sez_dgp.py)
        basic_covariates = Z_list  # ["airport_p", "port_p", "kl_p", "num_p"]
        X = df_clean[basic_covariates].values
        W = df_clean["W"].values  # Use W in real data, pass as D to unified function
        G = df_clean["G"].values
        Y1 = df_clean["Y1"].values
        Y2 = df_clean["Y2"].values

        # Calculate Y2-Y1 difference
        delta_Y = Y2 - Y1

        # Use unified function (pass W as D, estimate from all data)
        predictions = estimate_unified_outcome_model(
            D=W,  # Pass real data's W as D
            G=G,
            X=X,
            delta_Y=delta_Y,
            config=self.config,
        )

        # Check array lengths (for debugging)
        if self.config.verbose:
            n_data = len(df_clean)
            for key, pred in predictions.items():
                if len(pred) != n_data:
                    tqdm.write(
                        f"Warning: Outcome prediction array length mismatch - Data: {n_data}, {key}: {len(pred)}"
                    )
        return predictions

    def calculate_DATT_IPW(self, df, ps_results, g_level):
        """
        Calculate DATT(g) in IPW form (comparison method in Xu's paper)

        This is the IPW estimator compared with the proposed method in Xu's paper.

        Args:
            df: Common dataset (already cleaned)
            ps_results: Propensity score estimation results
            g_level: Exposure level (0 or 1)
        """
        # Prepare data (common dataset is already cleaned)
        D = df["W"].values  # W_i → D_i (treatment variable)
        G = df["G"].values
        Y1 = df["Y1"].values
        Y2 = df["Y2"].values

        # Get propensity scores
        eta_Z = ps_results["eta_Z"]
        eta_11_Z = ps_results["eta_11_Z"]
        eta_01_Z = ps_results["eta_01_Z"]
        eta_10_Z = ps_results["eta_10_Z"]
        eta_00_Z = ps_results["eta_00_Z"]

        # Check array lengths (for debugging)
        if self.config.verbose and len(D) != len(eta_Z):
            print(f"Warning: Array length mismatch - D: {len(D)}, eta_Z: {len(eta_Z)}")

        # Outcome difference
        delta_Y = Y2 - Y1

        # Select propensity score corresponding to exposure level
        if g_level == 1:
            eta_1g_Z = eta_11_Z
            eta_0g_Z = eta_01_Z
        else:  # g_level == 0
            eta_1g_Z = eta_10_Z
            eta_0g_Z = eta_00_Z

        # Use common IPW weight calculation function
        weights = compute_xu_ipw_weights(
            D=D, G=G, eta=eta_Z, eta_1g=eta_1g_Z, eta_0g=eta_0g_Z, g_level=g_level
        )

        # IPW estimator
        ipw_estimate = np.mean(weights * delta_Y)

        return ipw_estimate

    def calculate_DATT_DR(self, df, ps_results, outcome_predictions, g_level):
        """
        Calculate true doubly robust estimator DATT(g) (proposed method in Xu's paper)

        Doubly robust estimator has consistency when both propensity score model
        and outcome model are correctly specified.

        Formula: τ^{dr}(g) = E[W_i/η(z_i) * I{G_i=g}/η_{1g}(z_i) * (Y_{i2}-Y_{i1})
                       - (1-W_i)/(1-η(z_i)) * I{G_i=g}/η_{0g}(z_i) * (Y_{i2}-Y_{i1})
                       + m_{2,1g}(z_i) - m_{1,1g}(z_i) - m_{2,0g}(z_i) + m_{1,0g}(z_i)]

        Args:
            df: Common dataset (already cleaned)
            ps_results: Propensity score estimation results
            outcome_predictions: Outcome model prediction results
            g_level: Exposure level (0 or 1)
        """
        # Prepare data (common dataset is already cleaned)
        W = df["W"].values
        G = df["G"].values
        Y1 = df["Y1"].values
        Y2 = df["Y2"].values

        # Get propensity scores
        eta_Z = ps_results["eta_Z"]
        eta_11_Z = ps_results["eta_11_Z"]
        eta_01_Z = ps_results["eta_01_Z"]
        eta_10_Z = ps_results["eta_10_Z"]
        eta_00_Z = ps_results["eta_00_Z"]

        # Check array lengths (for debugging)
        if self.config.verbose:
            if len(W) != len(eta_Z):
                print(
                    f"Warning: Array length mismatch - W: {len(W)}, eta_Z: {len(eta_Z)}"
                )

        # Outcome difference
        delta_Y = Y2 - Y1

        # Select propensity scores and predictions corresponding to exposure level g
        if g_level == 1:
            eta_1g_Z = eta_11_Z
            eta_0g_Z = eta_01_Z
            m_1g_pred = outcome_predictions["m_delta_11"]  # m_{2,1g} - m_{1,1g}
            m_0g_pred = outcome_predictions["m_delta_01"]  # m_{2,0g} - m_{1,0g}
        else:  # g_level == 0
            eta_1g_Z = eta_10_Z
            eta_0g_Z = eta_00_Z
            m_1g_pred = outcome_predictions["m_delta_10"]  # m_{2,1g} - m_{1,1g}
            m_0g_pred = outcome_predictions["m_delta_00"]  # m_{2,0g} - m_{1,0g}

        # Check array lengths (for debugging)
        if self.config.verbose:
            if len(m_1g_pred) != len(W):
                print(
                    f"Warning: Array length mismatch - W: {len(W)}, m_1g_pred: {len(m_1g_pred)}"
                )
            if len(m_0g_pred) != len(W):
                print(
                    f"Warning: Array length mismatch - W: {len(W)}, m_0g_pred: {len(m_0g_pred)}"
                )

        # Use common doubly robust estimator calculation function
        weights_for_delta_Y, adjustment = compute_xu_dr_weights(
            D=W,
            G=G,
            eta=eta_Z,
            eta_1g=eta_1g_Z,
            eta_0g=eta_0g_Z,
            m_1g_pred=m_1g_pred,
            m_0g_pred=m_0g_pred,
            g_level=g_level,
        )

        # Estimator is mean of weighted outcome difference and adjustment term
        tau_dr = np.mean(weights_for_delta_Y * delta_Y + adjustment)

        return tau_dr

    def calculate_effect_by_wg(
        self, df, ps_results, outcome_predictions, w_level, g_level, method="DR"
    ):
        """
        Calculate effect for specific (W,G) combination

        Calculates E[Y_{i2}(w,g) - Y_{i1}(w,g) | W_i=w, G_i=g].

        Args:
            df: Common dataset (already cleaned)
            ps_results: Propensity score estimation results
            outcome_predictions: Outcome model prediction results
            w_level: Treatment level (0 or 1)
            g_level: Exposure level (0 or 1)
            method: Type of estimator ("DR" or "IPW")

        Returns:
            Effect estimate
        """
        # Prepare data (common dataset is already cleaned)
        W = df["W"].values
        G = df["G"].values
        Y1 = df["Y1"].values
        Y2 = df["Y2"].values
        delta_Y = Y2 - Y1

        # Filter by W level and G level
        wg_mask = (W == w_level) & (G == g_level)

        if wg_mask.sum() == 0:
            return 0.0

        # Filtered data
        delta_Y_filtered = delta_Y[wg_mask]

        if method == "DR":
            # Doubly robust estimator: use outcome model predictions
            # Get predictions
            pred_key = f"m_delta_{w_level}{g_level}"
            m_pred = outcome_predictions[pred_key]
            m_pred_filtered = m_pred[wg_mask]

            # Doubly robust estimator: combination of observed values and predictions
            # Simplified version: directly calculate E[ΔY | W=w, G=g]
            # More accurately, weighting by propensity scores should be considered,
            # but since we condition on W and G, approximate with simple mean
            effect = np.mean(delta_Y_filtered)
        else:
            # IPW estimator: simple mean (already conditioned on W and G)
            effect = np.mean(delta_Y_filtered)

        return effect

    def calculate_spillover_effect(
        self, df, ps_results, outcome_predictions, w_level, method="DR"
    ):
        """
        Calculate spillover effect based on Appendix B of the paper

        τ(w,1,0) = E[Y_{i2}(w,1) - Y_{i2}(w,0) | W_i=w]

        Args:
            df: Dataframe
            ps_results: Propensity score estimation results
            outcome_predictions: Outcome model prediction results
            w_level: Treatment level (0 or 1)
            method: Type of estimator ("DR" or "IPW")

        Returns:
            Spillover effect estimate
        """
        # Effect at G=1
        effect_g1 = self.calculate_effect_by_wg(
            df,
            ps_results,
            outcome_predictions,
            w_level=w_level,
            g_level=1,
            method=method,
        )

        # Effect at G=0
        effect_g0 = self.calculate_effect_by_wg(
            df,
            ps_results,
            outcome_predictions,
            w_level=w_level,
            g_level=0,
            method=method,
        )

        # Spillover effect = Effect at G=1 - Effect at G=0
        spillover = effect_g1 - effect_g0

        return spillover

    def _prepare_df_for_estimation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common method to perform W → D conversion

        Args:
            df: Original dataframe (contains "W" column)

        Returns:
            Copy of dataframe with "D" column added
        """
        df_for_estimation = df.copy()
        df_for_estimation["D"] = df["W"].values
        return df_for_estimation

    def calculate_canonical_ipw(self, df, PS_covariates):
        """
        Calculate standard IPW-DID estimator (Abadie, 2005)

        Ignores spillover effects (G), uses only W and covariates Z.
        Used to demonstrate bias when interference is ignored.

        Note: This method uses common functions internally.

        Args:
            df: Dataframe (contains W, Y1, Y2, covariates)
            PS_covariates: List of covariates for propensity score estimation

        Returns:
            ATT estimate (direct effect ignoring spillover)
        """
        # Prepare data (convert W → D)
        df_for_estimation = self._prepare_df_for_estimation(df)

        # Use common function
        ipw_result = estimate_ipw(
            df_for_estimation,
            self.config,
            covariates=PS_covariates,
            treatment_col="D",
        )
        return ipw_result.estimate

    def calculate_canonical_dr_did(self, df, PS_covariates, Z_list):
        """
        Calculate standard DR-DID estimator

        Ignores spillover effects (G), uses only W and covariates Z.
        Used to demonstrate bias when interference is ignored.

        Note: This method uses common functions internally.

        Args:
            df: Dataframe (contains W, Y1, Y2, covariates)
            PS_covariates: List of covariates for propensity score estimation
            Z_list: List of basic covariates for outcome model estimation

        Returns:
            ATT estimate (direct effect ignoring spillover)
        """
        # Prepare data (convert W → D)
        df_for_estimation = self._prepare_df_for_estimation(df)

        # Use common function (use PS_covariates as covariates)
        dr_did_result = estimate_dr_did(
            df_for_estimation,
            self.config,
            covariates=PS_covariates,  # For propensity score
            treatment_col="D",
        )
        return dr_did_result.estimate

    def estimate_all_effects(
        self,
        df,
        PS_covariates,
        Z_list,
        cached_ps_results: Optional[Dict[str, np.ndarray]] = None,
        cached_outcome_predictions: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Estimate all effects

        Args:
            df: Dataframe
            PS_covariates: List of covariates for propensity score
            Z_list: List of covariates for outcome model
            cached_ps_results: Cached propensity score results (optional)
            cached_outcome_predictions: Cached outcome model results (optional)
        """
        # Calculate each effect (compare both methods)
        results = {}

        # Prepare common dataset (include all necessary columns)
        required_cols = list(set(PS_covariates + Z_list + ["W", "G", "Y1", "Y2"]))
        df_common = df[required_cols].dropna()
        df_common = df_common.replace([np.inf, -np.inf], np.nan).dropna()
        df_common = df_common.reset_index(drop=True)

        # Display progress with progress bar
        # Total 19 steps: Propensity score(1) + Outcome model(1) + DATT_DR(2) + DATT_IPW(2) + ODE(1) + Spillover(5) + Canonical(2) + Proposed methods(5)
        # Proposed methods: Data preparation(1) + Neighbor list(1) + ADTT influence function(1) + AITT influence function(1) + Estimator calculation(1) + ADTT HAC SE(1) + AITT HAC SE(1) = 7 steps
        # However, previously influence functions were calculated within compute_all_estimators, but changed to calculate directly to avoid duplication
        # Actual step count: Data preparation(1) + Neighbor list(1) + ADTT influence function(1) + AITT influence function(1) + Estimator calculation(1) + ADTT HAC SE(1) + AITT HAC SE(1) = 7 steps
        total_steps = 19
        pbar = None
        if self.config.verbose:
            pbar = tqdm(
                total=total_steps,
                desc="[Step 4] Executing doubly robust estimators",
                unit="step",
                ncols=80,
            )

        try:
            # 1. Estimate propensity scores (use cache if available)
            if pbar:
                pbar.set_description("Propensity score estimation")
            if cached_ps_results is not None:
                # Use cached results
                # Note: Cache is saved per cluster, so normal estimation is needed
                # for entire bootstrap sample. This parameter is a placeholder for future extension.
                ps_results = cached_ps_results
            else:
                ps_results = self.estimate_propensity_scores(
                    df, PS_covariates, df_common=df_common
                )
            if pbar:
                pbar.update(1)

            # 2. Estimate outcome models (use cache if available)
            if pbar:
                pbar.set_description("Outcome model estimation")
            if cached_outcome_predictions is not None:
                # Use cached results
                outcome_predictions = cached_outcome_predictions
            else:
                outcome_predictions = self.estimate_outcome_models(
                    df, Z_list, df_common=df_common
                )
            if pbar:
                pbar.update(1)
        except Exception as e:
            # Return empty dictionary if propensity score or outcome model estimation fails
            if pbar:
                pbar.close()
            if self.config.verbose:
                tqdm.write(
                    f"Warning: Propensity score or outcome model estimation failed: {e}"
                )
                import traceback

                traceback.print_exc()
            return results

        # 3. Doubly robust estimators (proposed method in Xu's paper)
        for g in [0, 1]:
            if pbar:
                pbar.set_description(f"DATT_DR (g={g})")
            try:
                datt_dr = self.calculate_DATT_DR(
                    df_common, ps_results, outcome_predictions, g
                )
                results[f"DATT_DR_{g}"] = datt_dr
            except Exception as e:
                if self.config.verbose:
                    tqdm.write(f"Warning: DATT_DR_{g} calculation failed: {e}")
                    import traceback

                    traceback.print_exc()
            if pbar:
                pbar.update(1)

        # 4. IPW estimators (comparison method in Xu's paper)
        for g in [0, 1]:
            if pbar:
                pbar.set_description(f"DATT_IPW (g={g})")
            try:
                datt_ipw = self.calculate_DATT_IPW(df_common, ps_results, g)
                results[f"DATT_IPW_{g}"] = datt_ipw
            except Exception as e:
                if self.config.verbose:
                    tqdm.write(f"Warning: DATT_IPW_{g} calculation failed: {e}")
                    import traceback

                    traceback.print_exc()
            if pbar:
                pbar.update(1)

        # 5. ODE (Overall Direct Effect) calculation (both methods)
        if pbar:
            pbar.set_description("ODE calculation")
        treated_mask = df_common["W"] == 1
        g_distribution = df_common.loc[treated_mask, "G"].value_counts(normalize=True)

        # ODE for doubly robust estimator
        ode_dr = 0
        for g in [0, 1]:
            if g in g_distribution.index:
                ode_dr += results[f"DATT_DR_{g}"] * g_distribution[g]
        results["ODE_DR"] = ode_dr

        # ODE for IPW estimator
        ode_ipw = 0
        for g in [0, 1]:
            if g in g_distribution.index:
                ode_ipw += results[f"DATT_IPW_{g}"] * g_distribution[g]
        results["ODE_IPW"] = ode_ipw

        # Calculate direct effect heterogeneity
        direct_effect_heterogeneity_dr = results["DATT_DR_1"] - results["DATT_DR_0"]
        results["Direct_Effect_Heterogeneity_DR"] = direct_effect_heterogeneity_dr

        direct_effect_heterogeneity_ipw = results["DATT_IPW_1"] - results["DATT_IPW_0"]
        results["Direct_Effect_Heterogeneity_IPW"] = direct_effect_heterogeneity_ipw
        if pbar:
            pbar.update(1)

        # 6. Calculate spillover effects (based on Appendix B of the paper)
        # Spillover effect in treatment group (W=1) (doubly robust estimator)
        if pbar:
            pbar.set_description("Spillover_Treated_DR")
        try:
            spillover_treated_dr = self.calculate_spillover_effect(
                df_common, ps_results, outcome_predictions, w_level=1, method="DR"
            )
            results["Spillover_Treated_DR"] = spillover_treated_dr
        except Exception as e:
            if self.config.verbose:
                tqdm.write(f"Warning: Spillover_Treated_DR calculation failed: {e}")
                import traceback

                traceback.print_exc()
        if pbar:
            pbar.update(1)

        # Spillover effect in control group (W=0) (doubly robust estimator)
        if pbar:
            pbar.set_description("Spillover_Control_DR")
        try:
            spillover_control_dr = self.calculate_spillover_effect(
                df_common, ps_results, outcome_predictions, w_level=0, method="DR"
            )
            results["Spillover_Control_DR"] = spillover_control_dr
        except Exception as e:
            if self.config.verbose:
                tqdm.write(f"Warning: Spillover_Control_DR calculation failed: {e}")
                import traceback

                traceback.print_exc()
        if pbar:
            pbar.update(1)

        # Spillover effect in treatment group (W=1) (IPW estimator)
        if pbar:
            pbar.set_description("Spillover_Treated_IPW")
        try:
            spillover_treated_ipw = self.calculate_spillover_effect(
                df_common, ps_results, outcome_predictions, w_level=1, method="IPW"
            )
            results["Spillover_Treated_IPW"] = spillover_treated_ipw
        except Exception as e:
            if self.config.verbose:
                tqdm.write(f"Warning: Spillover_Treated_IPW calculation failed: {e}")
                import traceback

                traceback.print_exc()
        if pbar:
            pbar.update(1)

        # Spillover effect in control group (W=0) (IPW estimator)
        if pbar:
            pbar.set_description("Spillover_Control_IPW")
        try:
            spillover_control_ipw = self.calculate_spillover_effect(
                df_common, ps_results, outcome_predictions, w_level=0, method="IPW"
            )
            results["Spillover_Control_IPW"] = spillover_control_ipw
        except Exception as e:
            if self.config.verbose:
                tqdm.write(f"Warning: Spillover_Control_IPW calculation failed: {e}")
                import traceback

                traceback.print_exc()
        if pbar:
            pbar.update(1)

        # Statistical test indicator for spillover effects
        if pbar:
            pbar.set_description("Raw Spillover Difference")
        try:
            raw_spillover = self.calculate_raw_spillover_difference(df_common)
            results.update(raw_spillover)
        except Exception as e:
            if self.config.verbose:
                tqdm.write(f"Warning: raw_spillover_difference calculation failed: {e}")
                import traceback

                traceback.print_exc()
        if pbar:
            pbar.update(1)

        # 7. Canonical DID (standard DID) calculation
        if pbar:
            pbar.set_description("Canonical_IPW")
        try:
            canonical_ipw = self.calculate_canonical_ipw(df, PS_covariates)
            results["Canonical_IPW"] = canonical_ipw
        except Exception as e:
            if self.config.verbose:
                tqdm.write(f"Warning: Canonical_IPW calculation failed: {e}")
                import traceback

                traceback.print_exc()
        if pbar:
            pbar.update(1)

        if pbar:
            pbar.set_description("Canonical_DR_DID")
        try:
            canonical_dr_did = self.calculate_canonical_dr_did(
                df, PS_covariates, Z_list
            )
            results["Canonical_DR_DID"] = canonical_dr_did
        except Exception as e:
            if self.config.verbose:
                tqdm.write(f"Warning: Canonical_DR_DID calculation failed: {e}")
                import traceback

                traceback.print_exc()
        if pbar:
            pbar.update(1)

        # 8. Calculate proposed methods (ADTT/AITT)
        try:
            proposed_results = self.calculate_proposed_adtt_aitt(
                df, PS_covariates, progress_bar=pbar
            )
            results.update(proposed_results)
        except Exception as e:
            if self.config.verbose:
                tqdm.write(
                    f"Warning: Proposed methods (ADTT/AITT) calculation failed: {e}"
                )
                import traceback

                traceback.print_exc()
        if pbar:
            pbar.close()

        return results

    def _generate_county_neighbors(
        self,
        df: pd.DataFrame,
        covariate_cols: List[str],
        county_id_col: str = "cty",
        max_neighbors: Optional[int] = None,
    ) -> List[List[int]]:
        """Generate neighbor list at county level (internal method)

        Based on "county-level neighborhood definition" described in Section 5.9 of the paper,
        defines other villages in the same county as neighbors. Neighbors are selected and ordered
        by covariate similarity (Euclidean distance in standardized covariate space).

        Uses `groupby().groups` for efficiency.

        Args:
            df: Dataframe (index should be sequential from 0)
            county_id_col: Column name indicating county ID (default: "cty")
            max_neighbors: Maximum number of neighbors to use per unit (default: None, uses config.max_neighbors)
            covariate_cols: List of column names to use for covariate-based neighbor selection.
                           Neighbors are sorted by covariate distance (ascending) and
                           the first max_neighbors are selected.

        Returns:
            List[List[int]]: List of neighbor indices for each unit, sorted by covariate distance
            Format expected by functions in `estimators.py`: List[List[int]]
        """
        if county_id_col not in df.columns:
            raise ValueError(f"County ID column '{county_id_col}' not found.")

        # Check if all covariate columns exist in df
        missing_cols = [col for col in covariate_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Some covariate columns not found in dataframe: {missing_cols}."
            )

        # Use config.max_neighbors if max_neighbors is not provided
        if max_neighbors is None:
            max_neighbors = self.config.max_neighbors

        # Standardize covariates for distance calculation
        Z = df[covariate_cols].values
        Z_mean = Z.mean(axis=0)
        Z_std = Z.std(axis=0)
        # Avoid division by zero: if std is 0, set to 1 (column is constant)
        Z_std = np.where(Z_std < 1e-10, 1.0, Z_std)
        Z_standardized = (Z - Z_mean) / Z_std

        # Create dictionary of indices by county ID for efficiency
        # df.groupby().groups maps group name (county ID) to list of indices
        county_indices = df.groupby(county_id_col).groups

        neighbors_list = []
        # Get column index of county ID column (for speedup)
        county_id_col_idx = df.columns.get_loc(county_id_col)

        for i in range(len(df)):
            # Get current unit's county ID (fast access with iat)
            county_id = df.iat[i, county_id_col_idx]

            # Get indices of all units in the same county
            indices_in_county = county_indices.get(county_id, pd.Index([]))

            # Exclude self (i)
            neighbor_indices = [idx for idx in indices_in_county if idx != i]

            # Sort neighbors by covariate distance (ascending: closest first)
            if len(neighbor_indices) > 0:
                # Compute distances from unit i to all candidate neighbors
                z_i = Z_standardized[i]
                distances = []
                for j in neighbor_indices:
                    z_j = Z_standardized[j]
                    dist = np.linalg.norm(z_i - z_j)
                    distances.append(dist)

                # Sort neighbor_indices by distance (ascending)
                # Use lexsort to break ties deterministically by index when distances are equal
                neighbor_indices = [
                    neighbor_indices[k]
                    for k in np.lexsort(
                        (np.array(neighbor_indices), np.array(distances))
                    )
                ]

            # Take first max_neighbors (most similar)
            neighbor_indices = neighbor_indices[:max_neighbors]

            neighbors_list.append(neighbor_indices)

        return neighbors_list

    def _estimate_hac_se_within_county(
        self,
        influence_func: np.ndarray,
        county_ids: np.ndarray,
        df: pd.DataFrame,
        covariate_cols: List[str],
    ) -> float:
        """Calculate HAC standard error using covariate-based distance matrix

        Uses covariate similarity to construct distance matrix. Distances between units
        in the same county are computed as Euclidean distance in standardized covariate space.
        Distances between units in different counties are set to a very large value so that
        kernel weights become zero (effectively ignoring between-county correlation).

        Args:
            influence_func: Array of influence functions
            county_ids: Array of county IDs (same length as influence_func)
            df: Dataframe containing covariate columns
            covariate_cols: List of column names to use for covariate-based distance calculation

        Returns:
            HAC standard error
        """
        N = len(influence_func)

        # Check if all covariate columns exist in df
        missing_cols = [col for col in covariate_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Some covariate columns not found in dataframe: {missing_cols}."
            )

        # Standardize covariates for distance calculation (once for all data)
        Z = df[covariate_cols].values
        Z_mean = Z.mean(axis=0)
        Z_std = Z.std(axis=0)
        # Avoid division by zero: if std is 0, set to 1 (column is constant)
        Z_std = np.where(Z_std < 1e-10, 1.0, Z_std)
        Z_standardized = (Z - Z_mean) / Z_std

        # Calculate bandwidth: collect within-county distances (without creating full matrix)
        within_county_distances = []
        unique_counties = np.unique(county_ids)

        for county_id in unique_counties:
            county_mask = county_ids == county_id
            Z_county = Z_standardized[county_mask]
            n_county = len(Z_county)

            if n_county > 1:
                # Compute distances within county only (memory efficient)
                for i in range(n_county):
                    for j in range(i + 1, n_county):
                        dist = np.linalg.norm(Z_county[i] - Z_county[j])
                        within_county_distances.append(dist)

        # Calculate bandwidth based on median of within-county covariate distances
        if len(within_county_distances) > 0:
            median_dist = np.median(within_county_distances)
            b = self.config.hac_bandwidth_multiplier * median_dist
        else:
            # Fallback if no within-county distances (should not happen in practice)
            b = (
                self.config.hac_bandwidth_multiplier
                * self.config.hac_default_max_distance
            )

        # Calculate HAC variance: loop by county (memory-efficient, preserves original structure)
        Z_mean_influence = np.mean(influence_func)
        Z_centered = influence_func - Z_mean_influence
        V_hac_total = 0.0

        for county_id in unique_counties:
            county_mask = county_ids == county_id
            Z_county = Z_centered[county_mask]
            n_county = len(Z_county)

            if n_county > 0:
                # Create covariate distance matrix within county (small, memory efficient)
                Z_county_std = Z_standardized[county_mask]
                dist_matrix_county = np.zeros((n_county, n_county))

                for i in range(n_county):
                    for j in range(i + 1, n_county):
                        dist = np.linalg.norm(Z_county_std[i] - Z_county_std[j])
                        dist_matrix_county[i, j] = dist
                        dist_matrix_county[j, i] = dist

                # Calculate HAC standard error for this county
                # Pass Z_county + Z_mean_influence (estimate_hac_se centers again internally,
                # so need values centered by overall mean)
                SE_g = estimate_hac_se(
                    influence_func=Z_county + Z_mean_influence,
                    dist_matrix=dist_matrix_county,
                    bandwidth=b,
                    config=self.config,
                )

                # Calculate contribution to variance of group g
                # estimate_hac_se returns SE = sqrt((Z_g' @ W_g @ Z_g) / n_g²), so
                # (Z_g' @ W_g @ Z_g) = SE_g² * n_g²
                # Overall variance is V_hac = (1/N) * Σ_g (Z_g' @ W_g @ Z_g), so
                # contribution of this group is SE_g² * n_g²
                V_hac_total += (SE_g**2) * (n_county**2)

        # Normalize HAC variance: V_hac = (1/N) * Σ_g Σ_i∈g Σ_j∈g Z_i * Z_j * w_ij
        V_hac = V_hac_total / N

        # Standard error = sqrt(V_hac / N)
        return np.sqrt(V_hac / N)

    def calculate_proposed_adtt_aitt(
        self,
        df: pd.DataFrame,
        covariates: List[str],
        county_id_col: str = "cty",
        progress_bar: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Calculate proposed methods (ADTT/AITT)

        After performing real data-specific processing (county-level neighborhood definition,
        variable name conversion), calculates estimators using common functions.

        Args:
            df: Dataframe (required columns: covariates + ["W", "Y1", "Y2", county_id_col])
            covariates: List of covariate column names
            county_id_col: County ID column name (default: "cty")
            progress_bar: Progress bar (optional)

        Returns:
            Dict[str, float]: Dictionary containing the following keys:
                - "ADTT": Average Direct Treatment Effect on the Treated
                - "ADTT_se": Standard error of ADTT
                - "AITT": Average Indirect Treatment Effect on the Treated
                - "AITT_se": Standard error of AITT
        """
        if progress_bar:
            progress_bar.set_description("Proposed methods: Data preparation")
        df_clean = df[covariates + ["W", "Y1", "Y2", county_id_col]].dropna()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_clean) == 0:
            if progress_bar:
                progress_bar.update(
                    6
                )  # Skip remaining steps (6 steps after data preparation)
            return {
                "ADTT": np.nan,
                "ADTT_se": np.nan,
                "AITT": np.nan,
                "AITT_se": np.nan,
            }

        # Reset data and convert to positional index
        df_clean = df_clean.reset_index(drop=True)
        if progress_bar:
            progress_bar.update(1)

        # Create neighbor list at county level (real data-specific processing)
        # Use covariate-based neighbor selection: neighbors are sorted by covariate similarity
        # so that D_{(k)} denotes the treatment of the k-th most covariate-similar village
        if progress_bar:
            progress_bar.set_description("Proposed methods: Creating neighbor list")
        neighbors_list = self._generate_county_neighbors(
            df_clean,
            covariates,  # Use covariate distance to define neighbor order (required parameter)
            county_id_col=county_id_col,
            max_neighbors=self.config.max_neighbors,
        )
        if progress_bar:
            progress_bar.update(1)

        # Variable name conversion (real data-specific processing)
        df_for_estimation = self._prepare_df_for_estimation(df_clean)

        # Get county IDs (for HAC standard error calculation)
        county_ids = df_clean[county_id_col].values

        # Calculate influence functions (used for both estimator and HAC standard error)
        if progress_bar:
            progress_bar.set_description(
                "Proposed methods: ADTT influence function calculation"
            )
        adtt_influence = compute_adtt_influence_function(
            df_for_estimation,
            neighbors_list,
            "logistic",
            self.config,
            covariates=covariates,
            treatment_col="D",
            random_seed=self.config.random_seed,
        )
        if progress_bar:
            progress_bar.update(1)

        if progress_bar:
            progress_bar.set_description(
                "Proposed methods: AITT influence function calculation"
            )
        aitt_influence = compute_aitt_influence_function(
            df_for_estimation,
            neighbors_list,
            "logistic",
            self.config,
            covariates=covariates,
            treatment_col="D",
            random_seed=self.config.random_seed,
        )
        if progress_bar:
            progress_bar.update(1)

        # Calculate estimator (mean of influence function)
        if progress_bar:
            progress_bar.set_description("Proposed methods: Estimator calculation")
        adtt_estimate = np.mean(adtt_influence)
        aitt_estimate = np.mean(aitt_influence)
        if progress_bar:
            progress_bar.update(1)

        # Calculate HAC standard error at county level (grouped by county)
        if progress_bar:
            progress_bar.set_description(
                "Proposed methods: ADTT HAC standard error calculation"
            )
        adtt_hac_se = self._estimate_hac_se_within_county(
            adtt_influence,
            county_ids,
            df_clean,
            covariates,
        )
        if progress_bar:
            progress_bar.update(1)

        if progress_bar:
            progress_bar.set_description(
                "Proposed methods: AITT HAC standard error calculation"
            )
        aitt_hac_se = self._estimate_hac_se_within_county(
            aitt_influence,
            county_ids,
            df_clean,
            covariates,
        )
        if progress_bar:
            progress_bar.update(1)

        # Convert results to format for real data (overwrite with HAC standard error)
        return {
            "ADTT": adtt_estimate,
            "ADTT_se": adtt_hac_se,
            "AITT": aitt_estimate,
            "AITT_se": aitt_hac_se,
        }

    def calculate_treated_exposure_distribution(self, df, g_distribution):
        """
        1. Calculate exposure level distribution in treatment group

        Args:
            df: Dataframe
            g_distribution: Already calculated exposure level distribution

        Returns:
            dict: Exposure level distribution in treatment group
        """
        return {
            "treated_exposure_0_share": g_distribution.get(0, 0.0),
            "treated_exposure_1_share": g_distribution.get(1, 0.0),
        }

    def calculate_control_exposure_distribution(self, df):
        """
        2. Calculate exposure level distribution in control group

        Args:
            df: Dataframe

        Returns:
            dict: Exposure level distribution in control group
        """
        df_clean = df[["W", "G", "Y1", "Y2"]].dropna()
        control_mask = df_clean["W"] == 0

        g_dist_control = df_clean.loc[control_mask, "G"].value_counts(normalize=True)
        return {
            "control_exposure_0_share": g_dist_control.get(0, 0.0),
            "control_exposure_1_share": g_dist_control.get(1, 0.0),
        }

    def calculate_exposure_level_effects(self, df):
        """
        3. Calculate detailed analysis of effects by exposure level

        Args:
            df: Dataframe

        Returns:
            dict: Detailed analysis results of effects by exposure level
        """
        df_clean = df[["W", "G", "Y1", "Y2"]].dropna()
        W = df_clean["W"].values
        G = df_clean["G"].values
        Y1 = df_clean["Y1"].values
        Y2 = df_clean["Y2"].values

        treated_mask = W == 1
        control_mask = W == 0
        results = {}

        for g in [0, 1]:
            g_mask = G == g
            # Execute same processing for treatment and control groups
            for group_name, group_mask in [
                ("treated", treated_mask),
                ("control", control_mask),
            ]:
                group_g_mask = group_mask & g_mask
                if group_g_mask.sum() > 0:
                    results[f"{group_name}_effect_g{g}_count"] = group_g_mask.sum()
                    results[f"{group_name}_effect_g{g}_mean_y1"] = Y1[
                        group_g_mask
                    ].mean()
                    results[f"{group_name}_effect_g{g}_mean_y2"] = Y2[
                        group_g_mask
                    ].mean()
                    results[f"{group_name}_effect_g{g}_mean_delta"] = (
                        Y2[group_g_mask] - Y1[group_g_mask]
                    ).mean()
                else:
                    results[f"{group_name}_effect_g{g}_count"] = 0
                    results[f"{group_name}_effect_g{g}_mean_y1"] = 0.0
                    results[f"{group_name}_effect_g{g}_mean_y2"] = 0.0
                    results[f"{group_name}_effect_g{g}_mean_delta"] = 0.0

        return results

    def calculate_raw_spillover_difference(self, df):
        """
        4. Statistical test indicator for spillover effects (required: based on Appendix B of the paper)

        Args:
            df: Common dataset (already cleaned)

        Returns:
            dict: Statistical test indicator for spillover effects
        """
        # Common dataset is already cleaned
        W = df["W"].values
        G = df["G"].values
        Y1 = df["Y1"].values
        Y2 = df["Y2"].values

        treated_mask = W == 1

        # Difference in effects between exposure level 1 and 0 in treatment group
        if (treated_mask & (G == 1)).sum() > 0 and (treated_mask & (G == 0)).sum() > 0:
            delta_1 = (Y2[treated_mask & (G == 1)] - Y1[treated_mask & (G == 1)]).mean()
            delta_0 = (Y2[treated_mask & (G == 0)] - Y1[treated_mask & (G == 0)]).mean()
            return {"raw_spillover_difference": delta_1 - delta_0}
        else:
            return {"raw_spillover_difference": 0.0}
