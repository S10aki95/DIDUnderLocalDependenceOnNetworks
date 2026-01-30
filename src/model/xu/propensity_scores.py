"""
Xu Estimator: Propensity Score Estimation

Common utility for estimating propensity scores used in Xu (2025) estimators.
"""

import pandas as pd
import numpy as np
from ...settings import Config
from ...utils import clip_ps, get_ml_model
from ..common.models import PropensityScoreResult


def estimate_xu_propensity_scores(
    X: pd.DataFrame, D: pd.Series, G: pd.Series, config: Config
) -> PropensityScoreResult:
    """Estimate propensity scores required for Xu estimators

    Estimates P(D=1|Z), P(G=g|D=0,Z), P(G=g|D=1,Z).

    Args:
        X: Covariate DataFrame
        D: Treatment variable Series
        G: Exposure level Series
        config: Configuration object

    Returns:
        Propensity score estimation results
    """
    # 1. P(D=1|Z)
    ps_d_model = get_ml_model("logistic", config).fit(X, D)
    eta = clip_ps(ps_d_model.predict_proba(X)[:, 1], config)

    # 2. P(G=g|D, Z)
    # Train with features combining D and Z
    X_with_D = X.copy()
    X_with_D["D"] = D.values
    model_G = get_ml_model("logistic", config).fit(X_with_D, G)

    # Predict separately for D=0 and D=1 cases
    X_D0 = X.copy()
    X_D0["D"] = 0
    X_D1 = X.copy()
    X_D1["D"] = 1

    probs_D0 = model_G.predict_proba(X_D0)
    probs_D1 = model_G.predict_proba(X_D1)

    # Store results in dictionary format
    ps_g_models = {}
    g_classes = sorted(G.unique())
    for d_val in [0, 1]:
        probs = probs_D0 if d_val == 0 else probs_D1
        ps_g_models[d_val] = {
            g: (
                clip_ps(probs[:, model_G.classes_.tolist().index(g)], config)
                if g in model_G.classes_
                else np.full(len(X), config.epsilon)
            )
            for g in g_classes
        }

    return PropensityScoreResult(
        eta=eta,
        eta_g=ps_g_models,
        g_classes=g_classes,
        ps_d_model=ps_d_model,
        ps_g_model=model_G,
    )
