"""
Visualization configuration and constants module

Manages estimator mappings, label definitions, colors, markers, and other settings.
"""


class VisualizationConfig:
    """Constants class for visualization configuration"""

    # Estimator mappings
    ADTT_ESTIMATORS = {
        "Proposed IPW (ADTT)": "proposed_adtt",
        "Proposed DR (ADTT)": "proposed_dr_adtt",
        "Xu DR (CS)": "xu_dr_cs_ode",
        "Xu DR (MO)": "xu_dr_mo_ode",
        "Xu DR (FM)": "xu_dr_fm_ode",
        "Xu IPW (CS)": "xu_ipw_cs_ode",
        "Xu IPW (MO)": "xu_ipw_mo_ode",
        "Xu IPW (FM)": "xu_ipw_fm_ode",
        "Canonical IPW": "canonical_ipw",
        "Canonical TWFE": "canonical_twfe",
        "Modified TWFE": "modified_twfe",
        "DR-DID": "dr_did",
    }

    # Mapping of estimator names used in actual data
    ACTUAL_ESTIMATOR_NAMES = {
        "Proposed IPW (ADTT)": "Proposed IPW ADTT (Logistic)",
        "Proposed DR (ADTT)": "Proposed DR ADTT (Logistic)",
        "Xu DR (CS)": "Xu DR (CS) - ODE",
        "Xu DR (MO)": "Xu DR (MO) - ODE",
        "Xu DR (FM)": "Xu DR (FM) - ODE",
        "Xu IPW (CS)": "Xu IPW (CS) - ODE",
        "Xu IPW (MO)": "Xu IPW (MO) - ODE",
        "Xu IPW (FM)": "Xu IPW (FM) - ODE",
        "Canonical IPW": "Canonical IPW",
        "Canonical TWFE": "Canonical TWFE",
        "Modified TWFE": "Modified TWFE",
        "DR-DID": "DR-DID",
    }

    # Colors and markers for plots
    COLORS = ["blue", "green", "red", "orange", "purple", "brown", "pink"]
    MARKERS = ["o", "s", "^", "D", "v", "<", ">"]

    # English labels only
    LABELS = {
        "bias": "Bias",
        "rmse": "RMSE",
        "title_prefix": "Robustness Analysis",
        "sample_size": (
            "Sample Size (N)",
            "Sample Size vs Bias",
            "Sample Size vs RMSE",
        ),
        "density": (
            "Interference Range (K)",
            "Network Density vs Bias",
            "Network Density vs RMSE",
        ),
        "correlation": (
            "Spatial Correlation Strength",
            "Spatial Correlation vs Bias",
            "Spatial Correlation vs RMSE",
        ),
        "features": (
            "Number of Neighbor Features",
            "Neighbor Features vs Bias",
            "Neighbor Features vs RMSE",
        ),
        "spillover_x": "Number of Treated Neighbors ($S_i$)",
        "spillover_y": "Frequency",
        "spillover_title": "Figure 1: Distribution of Spillover Effect Counts",
        "units_x": "Spatial X-coordinate",
        "units_y": "Spatial Y-coordinate",
        "units_title": "Figure 3: Unit Locations and Treatment Assignment",
        "treated_label": "Treated Units (D=1)",
        "control_label": "Control Units (D=0)",
        "distribution_title": "Figure 2: Estimator Distribution (ADTT)",
        "distribution_x": "Estimator",
        "distribution_y": "Estimated Treatment Effect",
    }


def get_labels(key: str, metric: str = None) -> str | tuple:
    """Common function to get English labels"""
    label = VisualizationConfig.LABELS.get(key, key)

    # If label is a tuple, return appropriate element based on metric
    if isinstance(label, tuple):
        if metric == "Bias" and len(label) >= 2:
            return label[1]  # Title for Bias
        elif metric == "RMSE" and len(label) >= 3:
            return label[2]  # Title for RMSE
        else:
            return label[0]  # X-axis label

    return label
