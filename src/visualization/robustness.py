"""
Robustness visualization functions module

Provides visualization functions for robustness experiments.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List

from .config import VisualizationConfig, get_labels
from .utils import extract_estimator_data, setup_plot_style, save_plot


def create_robustness_line_plot(
    results_data: Dict[str, pd.DataFrame],
    x_variable: str,
    x_values: list,
    metric: str,
    save_path: Optional[str] = None,
    language: str = "auto",
):
    """
    Create robustness analysis line plot

    Args:
        results_data: Dictionary of result dataframes for each setting
        x_variable: X-axis variable name
        x_values: List of X-axis values
        metric: Evaluation metric ("Bias" or "RMSE")
        save_path: Save path
        language: Language setting
    """
    # Generate CSV file before creating plot (for use in R scripts)
    if save_path:
        output_dir = os.path.dirname(save_path) or "results"
    else:
        output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)

    # Organize data
    # Plot only same estimators to match R scripts
    # Corresponds to estimator_mapping in R scripts
    estimator_mapping = {
        "Proposed IPW (ADTT)": "Proposed IPW ADTT (Logistic)",
        "Proposed DR (ADTT)": "Proposed DR ADTT (Logistic)",
        "Xu (Oracle)": "Xu DR (CS) - ODE",
        "Xu (MO)": "Xu DR (MO) - ODE",
        "Xu (FM)": "Xu DR (FM) - ODE",
        "Canonical IPW": "Canonical IPW",
        "Canonical TWFE": "Canonical TWFE",
        "DR-DID": "DR-DID",
        "Modified TWFE": "Modified TWFE",
    }

    plot_data = {}
    plot_csv_data = []
    for estimator_name, actual_name in estimator_mapping.items():
        values = extract_estimator_data(
            results_data, x_values, metric, estimator_name, actual_name
        )
        plot_data[estimator_name] = values

        # Prepare data for CSV
        for x_val, value in zip(x_values, values):
            plot_csv_data.append(
                {
                    "x_value": x_val,
                    "estimator": estimator_name,
                    "metric": metric,
                    "value": value,
                }
            )

    # Save CSV file (must be generated before creating plot)
    if plot_csv_data:
        plot_csv_path = os.path.join(
            output_dir, f"plot_data_robustness_{x_variable}_{metric.lower()}.csv"
        )
        plot_df = pd.DataFrame(plot_csv_data)
        plot_df.to_csv(plot_csv_path, index=False)
        print(f"Saved plot data: {plot_csv_path}")

    # Create plot
    setup_plot_style((10, 6))

    for i, (estimator_name, values) in enumerate(plot_data.items()):
        if any(v != 0 for v in values):  # Plot only if there is valid data
            plt.plot(
                x_values,
                values,
                marker=VisualizationConfig.MARKERS[
                    i % len(VisualizationConfig.MARKERS)
                ],
                color=VisualizationConfig.COLORS[i % len(VisualizationConfig.COLORS)],
                linewidth=2,
                markersize=6,
                label=estimator_name,
            )

    # Set labels
    ylabel = get_labels(metric.lower())
    title_prefix = get_labels("title_prefix")

    # Set X-axis label and title (to match R scripts)
    if "sample_size" in x_variable.lower():
        xlabel = get_labels("sample_size")
    elif "density" in x_variable.lower() or "k" in x_variable.lower():
        xlabel = get_labels("density")
    elif "correlation" in x_variable.lower():
        xlabel = get_labels("correlation")
    elif "features" in x_variable.lower():
        xlabel = get_labels("features")
    else:
        xlabel = x_variable

    # Same title format as R scripts: "Robustness Analysis: Bias vs. Interference Range (K)"
    title = f"{title_prefix}: {ylabel} vs. {xlabel}"

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set axis ranges appropriately
    if x_values and any(v != 0 for v in plot_data.get(list(plot_data.keys())[0], [])):
        plt.xlim(
            min(x_values) - (max(x_values) - min(x_values)) * 0.05,
            max(x_values) + (max(x_values) - min(x_values)) * 0.05,
        )

        all_y_values = []
        for values in plot_data.values():
            all_y_values.extend([v for v in values if v != 0])
        if all_y_values:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            y_range = y_max - y_min
            if y_range > 0:
                plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    plt.tight_layout()
    save_plot(save_path, "Figure")


def create_sensitivity_line_plot(
    results_data: Dict[str, pd.DataFrame],
    x_variable: str,
    x_values: list,
    metric: str,
    save_path: Optional[str] = None,
    language: str = "auto",
):
    """
    Create sensitivity analysis line plot (both Proposed IPW and DR (ADTT))

    Args:
        results_data: Dictionary of result dataframes for each setting (DataFrame loaded from CSV)
        x_variable: X-axis variable name
        x_values: List of X-axis values
        metric: Evaluation metric ("Bias" or "RMSE")
        save_path: Save path
        language: Language setting
    """
    # Generate CSV file before creating plot (for use in R scripts)
    if save_path:
        output_dir = os.path.dirname(save_path) or "results"
    else:
        output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)

    # Target both Proposed IPW and DR (ADTT)
    estimator_mapping = {
        "Proposed IPW (ADTT)": "Proposed IPW ADTT (Logistic)",
        "Proposed DR (ADTT)": "Proposed DR ADTT (Logistic)",
    }

    # Organize data
    plot_data = {}
    plot_csv_data = []
    for estimator_name, actual_name in estimator_mapping.items():
        values = extract_estimator_data(
            results_data, x_values, metric, estimator_name, actual_name
        )
        plot_data[estimator_name] = values

        # Prepare data for CSV
        for x_val, value in zip(x_values, values):
            plot_csv_data.append(
                {
                    "x_value": x_val,
                    "estimator": estimator_name,
                    "metric": metric,
                    "value": value,
                }
            )

    # Save CSV file (must be generated before creating plot)
    if plot_csv_data:
        plot_csv_path = os.path.join(
            output_dir, f"plot_data_sensitivity_{x_variable}_{metric.lower()}.csv"
        )
        plot_df = pd.DataFrame(plot_csv_data)
        plot_df.to_csv(plot_csv_path, index=False)
        print(f"Saved plot data: {plot_csv_path}")

    # Create plot
    setup_plot_style((10, 6))

    # Plot both IPW and DR
    colors = ["#2E86AB", "#1E5F8B"]  # Different colors for IPW and DR
    linestyles = ["solid", "dashed"]  # Different line styles for IPW and DR
    markers = ["o", "s"]  # Different markers for IPW and DR

    for i, (estimator_name, values) in enumerate(plot_data.items()):
        if any(v != 0 for v in values):  # Plot only if there is valid data
            plt.plot(
                x_values,
                values,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2,
                markersize=6,
                label=estimator_name,
            )

    # Set labels
    ylabel = get_labels(metric.lower())
    title_prefix = get_labels("title_prefix")
    xlabel = get_labels("features")

    # Same title format as R scripts: "Sensitivity Analysis: Bias vs. Number of Neighbor Features"
    title = f"Sensitivity Analysis: {ylabel} vs. {xlabel}"

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set axis ranges appropriately
    all_y_values = []
    for values in plot_data.values():
        all_y_values.extend([v for v in values if v != 0])

    if x_values and all_y_values:
        plt.xlim(
            min(x_values) - (max(x_values) - min(x_values)) * 0.05,
            max(x_values) + (max(x_values) - min(x_values)) * 0.05,
        )

        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min
        if y_range > 0:
            plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    plt.tight_layout()
    save_plot(save_path, "Figure")


def create_robustness_plots(
    all_results: Dict[str, Any],
    output_dir: str = "results",
    language: str = "auto",
):
    """
    Create all robustness analysis figures

    Args:
        all_results: Results of robustness experiments (format passed from main.py)
        output_dir: Output directory
        language: Language setting

    Note:
        This function generates `robustness_results.csv` from `all_results` before generating plots.
        Using the same data source as R scripts ensures numerical consistency.
    """
    # First, generate robustness_results.csv from all_results (must be generated before creating plots)
    robustness_data = []
    for experiment_id, result in all_results.items():
        if "error" in result or "evaluation_results" not in result:
            continue

        eval_data = result["evaluation_results"]
        for _, row in eval_data.iterrows():
            # Extract experiment type from experiment_id
            experiment_type = (
                experiment_id.split("_")[0] if "_" in experiment_id else "unknown"
            )
            # Map experiment_type (to match R scripts)
            if experiment_type == "sample":
                experiment_type = "sample"
            elif experiment_type == "network":
                experiment_type = "network"
            elif experiment_type == "spatial":
                experiment_type = "spatial"
            elif experiment_type == "neighbor":
                experiment_type = "neighbor"
            else:
                # Determine from experiment_id
                if "sample_size" in experiment_id:
                    experiment_type = "sample"
                elif "network_density" in experiment_id or "k_" in experiment_id:
                    experiment_type = "network"
                elif (
                    "spatial_correlation" in experiment_id
                    or "correlation" in experiment_id
                ):
                    experiment_type = "spatial"
                elif (
                    "neighbor_features" in experiment_id or "features" in experiment_id
                ):
                    experiment_type = "neighbor"
                else:
                    experiment_type = "unknown"

            robustness_data.append(
                {
                    "experiment_type": experiment_type,
                    "experiment_id": experiment_id,
                    "estimator": row["Estimator"],
                    "bias": row["Bias"],
                    "rmse": row["RMSE"],
                    "coverage_rate": row.get(
                        "Coverage_Rate", row.get("Coverage Rate", None)
                    ),
                    "config_name": result.get("config_name", "robustness"),
                    "overrides": (
                        str(result.get("overrides", {}))
                        if result.get("overrides")
                        else None
                    ),
                }
            )

    # Generate robustness_results.csv (must be generated before creating plots)
    if robustness_data:
        robustness_df = pd.DataFrame(robustness_data)
        robustness_csv_path = os.path.join(output_dir, "robustness_results.csv")
        robustness_df.to_csv(robustness_csv_path, index=False)
        print(f"✓ Generated robustness_results.csv: {robustness_csv_path}")
    else:
        print(f"Warning: No robustness data found. Skipping robustness plots.")
        return

    # Organize data by experiment type (from CSV)
    experiment_groups = {}
    for exp_type in ["sample_size", "density", "correlation", "features"]:
        experiment_groups[exp_type] = []

    # Determine experiment type from CSV
    if "experiment_type" in robustness_df.columns:
        for exp_type in robustness_df["experiment_type"].unique():
            if exp_type == "sample":
                experiment_groups["sample_size"].extend(
                    robustness_df[robustness_df["experiment_type"] == exp_type][
                        "experiment_id"
                    ].unique()
                )
            elif exp_type == "network":
                experiment_groups["density"].extend(
                    robustness_df[robustness_df["experiment_type"] == exp_type][
                        "experiment_id"
                    ].unique()
                )
            elif exp_type == "spatial":
                experiment_groups["correlation"].extend(
                    robustness_df[robustness_df["experiment_type"] == exp_type][
                        "experiment_id"
                    ].unique()
                )
            elif exp_type == "neighbor":
                experiment_groups["features"].extend(
                    robustness_df[robustness_df["experiment_type"] == exp_type][
                        "experiment_id"
                    ].unique()
                )

    # Sample size experiment
    if experiment_groups["sample_size"]:
        sample_size_data = {}
        sample_sizes = []

        for experiment_id in experiment_groups["sample_size"]:
            # Extract sample size from experiment ID (e.g., sample_size_200 -> 200)
            try:
                size = int(experiment_id.split("_")[-1])
                sample_sizes.append(size)
                # Extract data for corresponding experiment from CSV
                exp_data = robustness_df[
                    robustness_df["experiment_id"] == experiment_id
                ].copy()
                sample_size_data[size] = exp_data
            except (ValueError, IndexError):
                continue

        if sample_size_data and sample_sizes:
            sample_sizes.sort()

            # Figure 3: Relationship between sample size and Bias
            create_robustness_line_plot(
                sample_size_data,
                "sample_size",
                sample_sizes,
                "Bias",
                os.path.join(output_dir, "robustness_bias_vs_sample_size.png"),
                language,
            )

            # Figure 4: Relationship between sample size and RMSE
            create_robustness_line_plot(
                sample_size_data,
                "sample_size",
                sample_sizes,
                "RMSE",
                os.path.join(output_dir, "robustness_rmse_vs_sample_size.png"),
                language,
            )

    # Network density experiment
    if experiment_groups["density"]:
        density_data = {}
        k_values = []

        for experiment_id in experiment_groups["density"]:
            # Extract K value from experiment ID (e.g., network_density_0_8 -> 0.8)
            try:
                if "network_density" in experiment_id:
                    # network_density_0_8 -> 0.8
                    k_str = (
                        experiment_id.split("_")[-2]
                        + "."
                        + experiment_id.split("_")[-1]
                    )
                else:
                    # k_0_8 -> 0.8
                    k_str = (
                        experiment_id.split("_")[-2]
                        + "."
                        + experiment_id.split("_")[-1]
                    )
                k = float(k_str)
                k_values.append(k)
                # Extract data for corresponding experiment from CSV
                exp_data = robustness_df[
                    robustness_df["experiment_id"] == experiment_id
                ].copy()
                density_data[k] = exp_data
            except (ValueError, IndexError):
                continue

        if density_data and k_values:
            k_values.sort()

            # Figure 5: Relationship between network density and Bias
            create_robustness_line_plot(
                density_data,
                "density",
                k_values,
                "Bias",
                os.path.join(output_dir, "robustness_bias_vs_density.png"),
                language,
            )

            # Figure 6: Relationship between network density and RMSE
            create_robustness_line_plot(
                density_data,
                "density",
                k_values,
                "RMSE",
                os.path.join(output_dir, "robustness_rmse_vs_density.png"),
                language,
            )

    # Spatial correlation experiment
    if experiment_groups["correlation"]:
        correlation_data = {}
        corr_values = []

        for experiment_id in experiment_groups["correlation"]:
            # Extract correlation value from experiment ID (e.g., spatial_correlation_0_8 -> 0.8)
            try:
                if "spatial_correlation" in experiment_id:
                    # spatial_correlation_0_8 -> 0.8
                    corr_str = (
                        experiment_id.split("_")[-2]
                        + "."
                        + experiment_id.split("_")[-1]
                    )
                else:
                    # correlation_0_8 -> 0.8
                    corr_str = (
                        experiment_id.split("_")[-2]
                        + "."
                        + experiment_id.split("_")[-1]
                    )
                corr = float(corr_str)
                corr_values.append(corr)
                # Extract data for corresponding experiment from CSV
                exp_data = robustness_df[
                    robustness_df["experiment_id"] == experiment_id
                ].copy()
                correlation_data[corr] = exp_data
            except (ValueError, IndexError):
                continue

        if correlation_data and corr_values:
            corr_values.sort()

            # Figure 7: Relationship between spatial correlation and Bias
            create_robustness_line_plot(
                correlation_data,
                "correlation",
                corr_values,
                "Bias",
                os.path.join(output_dir, "robustness_bias_vs_correlation.png"),
                language,
            )

            # Figure 8: Relationship between spatial correlation and RMSE
            create_robustness_line_plot(
                correlation_data,
                "correlation",
                corr_values,
                "RMSE",
                os.path.join(output_dir, "robustness_rmse_vs_correlation.png"),
                language,
            )

    # Neighbor feature count experiment (Proposed IPW (ADTT) only)
    if experiment_groups["features"]:
        features_data = {}
        feature_values = []

        for experiment_id in experiment_groups["features"]:
            # Extract feature count from experiment ID (e.g., neighbor_features_10 -> 10)
            try:
                if "neighbor_features" in experiment_id:
                    # neighbor_features_10 -> 10
                    features = int(experiment_id.split("_")[-1])
                else:
                    # features_10 -> 10
                    features = int(experiment_id.split("_")[-1])
                feature_values.append(features)
                # Extract data for corresponding experiment from CSV
                exp_data = robustness_df[
                    robustness_df["experiment_id"] == experiment_id
                ].copy()
                features_data[features] = exp_data
            except (ValueError, IndexError):
                continue

        if features_data and feature_values:
            feature_values.sort()

            # Figure 9: Relationship between neighbor feature count and Bias (Proposed IPW (ADTT) only)
            create_sensitivity_line_plot(
                features_data,
                "features",
                feature_values,
                "Bias",
                os.path.join(output_dir, "sensitivity_bias_vs_features.png"),
                language,
            )

            # Figure 10: Relationship between neighbor feature count and RMSE (Proposed IPW (ADTT) only)
            create_sensitivity_line_plot(
                features_data,
                "features",
                feature_values,
                "RMSE",
                os.path.join(output_dir, "sensitivity_rmse_vs_features.png"),
                language,
            )
