"""
Common Pydantic models for estimator results

Estimators with the same purpose use common models.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any
import numpy as np


class ATTEstimateResult(BaseModel):
    """ATT estimator results

    Stores ATT (Average Treatment on Treated) estimate and standard error.
    """

    estimate: float = Field(description="ATT estimate")
    standard_error: float = Field(ge=0.0, description="Standard error (non-negative)")

    @field_validator("estimate", "standard_error")
    @classmethod
    def validate_finite(cls, v):
        """Check for NaN and infinity"""
        if np.isnan(v) or np.isinf(v):
            return v  # NaN is allowed (error handling is done by caller)
        return v

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.ndarray: lambda v: v.tolist(),
            np.float64: float,
            np.float32: float,
        },
    )


class XuEstimateResult(BaseModel):
    """Xu (2025) estimator results

    Stores direct effect DATT(g) for each exposure level and overall direct effect ODE.
    """

    datt_estimates: Dict[str, float] = Field(
        description="Dictionary of DATT(g) estimates for each exposure level g"
    )
    ode: float = Field(description="Overall Direct Effect (ODE) estimate")
    datt_standard_errors: Dict[str, float] = Field(
        description="Dictionary of DATT(g) standard errors for each exposure level g"
    )
    ode_standard_error: float = Field(
        ge=0.0, description="ODE standard error (non-negative)"
    )

    @field_validator("ode", "ode_standard_error")
    @classmethod
    def validate_finite(cls, v):
        """Check for NaN and infinity"""
        if np.isnan(v) or np.isinf(v):
            return v  # NaN is allowed (error handling is done by caller)
        return v

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.ndarray: lambda v: v.tolist(),
            np.float64: float,
            np.float32: float,
        },
    )


class PropensityScoreResult(BaseModel):
    """Propensity score estimation results

    Stores propensity score estimation results required for Xu (2025) estimators.
    """

    eta: np.ndarray = Field(description="Array of P(D=1|Z) estimates")
    eta_g: Dict[int, Dict[int, np.ndarray]] = Field(
        description="P(G=g|D=d,Z) estimates. Outer key is D value, inner key is G value"
    )
    g_classes: list = Field(description="List of exposure level classes")
    ps_d_model: Any = Field(description="Model object for P(D=1|Z)")
    ps_g_model: Any = Field(description="Model object for P(G=g|D,Z)")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.ndarray: lambda v: v.tolist(),
        },
    )
