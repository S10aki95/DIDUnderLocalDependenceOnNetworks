"""
Basic visualization functions module

Provides basic visualization functions for spillover, unit placement, distribution plots, etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from .config import VisualizationConfig, get_labels
from .utils import setup_plot_style, save_plot


def create_spillover_plot(
    save_path: Optional[str] = None,
    language: str = "auto",
    dgp_config: Optional[Any] = None,
    df: pd.DataFrame = None,
):
    """
    Figure 1: Distribution of spillover effect counts

    Args:
        save_path: Save path (display only if None)
        language: Language setting ("auto", "japanese", "english")
        dgp_config: DGP configuration (uses default if None)
        df: Dataframe (including S column). Required parameter. Pass data generated during simulation execution.

    Raises:
        ValueError: If df is None or not provided
    """
    # Verify that dataframe is required
    if df is None:
        raise ValueError(
            "create_spillover_plot: df parameter is required. "
            "Please pass the dataframe generated during simulation execution. "
            "Generating data separately may cause deviation from intended settings."
        )

    # Save data (before creating figure)
    if save_path:
        output_dir = os.path.dirname(save_path) or "results"
    else:
        output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, "plot_data_spillover.csv")

    # Save distribution data for S column
    if "S" in df.columns:
        s_counts = df["S"].value_counts().sort_index()
        plot_data = pd.DataFrame({"S": s_counts.index, "Frequency": s_counts.values})
        plot_data.to_csv(data_file, index=False)

    # Dynamically get spillover effects from DGP configuration
    if dgp_config is None:
        from src.settings import get_config

        config = get_config("default")
        dgp_config = config

    # Calculate distribution of S column (number of treated neighbors)
    S_counts = df["S"].value_counts().sort_index()
    S_values = S_counts.index.tolist()
    S_frequencies = S_counts.values.tolist()

    # Set labels
    xlabel = get_labels("spillover_x")
    ylabel = get_labels("spillover_y")
    title = get_labels("spillover_title")
    annotation_text = "(Spillover Effects)"

    setup_plot_style((10, 6))
    bars = plt.bar(
        S_values, S_frequencies, color="steelblue", alpha=0.7, edgecolor="black"
    )

    # Display value on each bar
    for bar, value in zip(bars, S_frequencies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(S_values)

    # Set axis ranges appropriately
    plt.xlim(min(S_values) - 0.5, max(S_values) + 0.5)
    plt.ylim(0, max(S_frequencies) * 1.1)

    # Add statistics
    total_units = len(df)
    stats_text = (
        f"Total Units: {total_units}\nMean Treated Neighbors: {df['S'].mean():.2f}"
    )

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    save_plot(save_path, "Figure")


def create_units_scatter_plot(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    language: str = "auto",
):
    """
    Figure 3: Scatter plot of unit placement and treatment status

    Args:
        df: Dataframe (including x, y, D columns)
        save_path: Save path (display only if None)
        language: Language setting ("auto", "japanese", "english")
    """
    # Save data (before creating figure)
    if save_path:
        output_dir = os.path.dirname(save_path) or "results"
    else:
        output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, "plot_data_units_scatter.csv")

    # Save unit placement data
    if all(col in df.columns for col in ["x", "y", "D"]):
        plot_data = df[["x", "y", "D"]].copy()
        plot_data.to_csv(data_file, index=False)

    # Set labels
    xlabel = get_labels("units_x")
    ylabel = get_labels("units_y")
    title = get_labels("units_title")
    treated_label = get_labels("treated_label")
    control_label = get_labels("control_label")

    setup_plot_style((10, 8))

    # Plot treated and control groups separately
    treated_mask = df["D"] == 1
    control_mask = df["D"] == 0

    # Control group (blue)
    plt.scatter(
        df.loc[control_mask, "x"],
        df.loc[control_mask, "y"],
        c="lightblue",
        alpha=0.7,
        s=30,
        label=control_label,
        edgecolors="blue",
        linewidth=0.5,
    )

    # Treated group (red)
    plt.scatter(
        df.loc[treated_mask, "x"],
        df.loc[treated_mask, "y"],
        c="red",
        alpha=0.8,
        s=50,
        label=treated_label,
        edgecolors="darkred",
        linewidth=0.5,
    )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Set axis ranges appropriately
    plt.xlim(0, df["x"].max() * 1.05)
    plt.ylim(0, df["y"].max() * 1.05)

    # Add statistical information
    n_treated = treated_mask.sum()
    n_control = control_mask.sum()
    n_total = len(df)
    treatment_rate = n_treated / n_total

    stats_text = f"Total Units: {n_total}\nTreated Units: {n_treated} ({treatment_rate:.1%})\nControl Units: {n_control} ({1-treatment_rate:.1%})"

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    save_plot(save_path, "Figure")


def create_distribution_plot_adtt(
    results_df: pd.DataFrame,
    true_params: Dict[str, float],
    save_path: Optional[str] = None,
    language: str = "auto",
):
    """
    Figure 2: Distribution of each estimator (ADTT only)

    Args:
        results_df: DataFrame of simulation results
        true_params: True parameter values
        save_path: Save path (if None, display only)
        language: Language setting ("auto", "japanese", "english")
    """
    # Save data (before creating figure)
    if save_path:
        output_dir = os.path.dirname(save_path) or "results"
    else:
        output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, "plot_data_adtt.csv")

    # Prepare data for figure creation
    plot_data = results_df.copy()
    estimator_cols = list(VisualizationConfig.ADTT_ESTIMATORS.values())
    available_cols = [col for col in estimator_cols if col in plot_data.columns]
    if "true_adtt" in plot_data.columns:
        available_cols.append("true_adtt")
    if "true_aitt" in plot_data.columns:
        available_cols.append("true_aitt")

    if available_cols:
        plot_data_subset = plot_data[available_cols]
        plot_data_subset.to_csv(data_file, index=False)

    # Select only valid estimators
    valid_estimators = {}
    for name, col in VisualizationConfig.ADTT_ESTIMATORS.items():
        if col in results_df.columns:
            valid_data = results_df[col].dropna()
            if len(valid_data) > 0:
                valid_estimators[name] = valid_data

    if not valid_estimators:
        print("Warning: No valid ADTT estimators found")
        return

    # Prepare plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    adtt_data = [v.values for v in valid_estimators.values()]
    adtt_labels = list(valid_estimators.keys())

    bp = ax.boxplot(adtt_data, labels=adtt_labels, patch_artist=True)

    # Color coding
    colors = [
        "lightblue",
        "lightgreen",
        "lightcoral",
        "lightyellow",
        "lightpink",
        "lightgray",
    ]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)

    # Display true ADTT value as red dashed line
    true_label = f'True Treatment Effect = {true_params["adtt"]:.3f}'
    title = get_labels("distribution_title")
    ylabel = get_labels("distribution_y")
    xlabel = get_labels("distribution_x")

    ax.axhline(
        y=true_params["adtt"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=true_label,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis ranges appropriately
    all_values = []
    for data in adtt_data:
        all_values.extend(data)
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    plt.tight_layout()
    save_plot(save_path, "Figure")


def create_distribution_plot_aitt(
    results_df: pd.DataFrame,
    true_params: Dict[str, float],
    save_path: Optional[str] = None,
    language: str = "auto",
):
    """
    Function to visualize distribution of AITT estimators

    Args:
        results_df: DataFrame of simulation results
        true_params: True parameter values
        save_path: Save path (if None, display only)
        language: Language setting ("auto", "japanese", "english")
    """
    # Save data (before creating figure)
    if save_path:
        output_dir = os.path.dirname(save_path) or "results"
    else:
        output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, "plot_data_aitt.csv")

    # Get AITT estimator data
    aitt_estimators = {}
    if "proposed_aitt" in results_df.columns:
        valid_data = results_df["proposed_aitt"].dropna()
        if len(valid_data) > 0:
            aitt_estimators["Proposed IPW (AITT)"] = valid_data

    if "proposed_dr_aitt" in results_df.columns:
        valid_data = results_df["proposed_dr_aitt"].dropna()
        if len(valid_data) > 0:
            aitt_estimators["Proposed DR (AITT)"] = valid_data

    if not aitt_estimators:
        print("Warning: No valid AITT data found")
        return

    # Save data (before creating figure)
    # Match column names expected by R script
    plot_data = pd.DataFrame()
    for name, data in aitt_estimators.items():
        if name == "Proposed IPW (AITT)":
            plot_data["proposed_aitt"] = data
        elif name == "Proposed DR (AITT)":
            plot_data["proposed_dr_aitt"] = data
        else:
            # Fallback: existing conversion logic
            col_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plot_data[col_name] = data
    if "true_aitt" in results_df.columns:
        plot_data["true_aitt"] = results_df["true_aitt"]
    if len(plot_data) > 0:
        plot_data.to_csv(data_file, index=False)

    # Prepare plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create box plot
    aitt_data = [v.values for v in aitt_estimators.values()]
    aitt_labels = list(aitt_estimators.keys())

    bp = ax.boxplot(aitt_data, labels=aitt_labels, patch_artist=True)

    # Color coding
    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)

    # Display true AITT value as red dashed line
    true_aitt = true_params.get("aitt", 0.0)
    true_label = f"True AITT = {true_aitt:.3f}"

    ax.axhline(
        y=true_aitt,
        color="red",
        linestyle="--",
        linewidth=2,
        label=true_label,
    )

    ax.set_title("Estimator Distribution (AITT)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Estimator", fontsize=12)
    ax.set_ylabel("Estimated AITT", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis ranges appropriately
    all_values = []
    for data in aitt_data:
        all_values.extend(data)
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    plt.tight_layout()
    save_plot(save_path, "Figure")


def create_coverage_rate_table(
    evaluation_results: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create Table 2: Coverage rate table for confidence intervals

    Args:
        evaluation_results: DataFrame of evaluation results
        config: Configuration dictionary (to get comparison target estimators)
        save_path: Save path (if None, do not save)

    Returns:
        pd.DataFrame: Coverage rate table
    """
    import numpy as np

    # Target estimators (obtained from config, default based on README.md requirements)
    if config and hasattr(config, "coverage_table_estimators"):
        target_estimators = config.coverage_table_estimators
    else:
        target_estimators = ["Proposed IPW (ADTT)", "Canonical IPW"]

    # Mapping of estimator names used in actual data
    actual_estimator_names = {
        "Proposed IPW (ADTT)": "Proposed IPW ADTT (Logistic)",
        "Proposed IPW (AITT)": "Proposed IPW AITT (Logistic)",
        "Proposed DR (ADTT)": "Proposed DR ADTT (Logistic)",
        "Proposed DR (AITT)": "Proposed DR AITT (Logistic)",
        "Canonical IPW": "Canonical IPW",
        "Canonical TWFE": "Canonical TWFE",
        "DR-DID": "DR-DID",
        "Modified TWFE": "Modified TWFE",
        "Xu DR (CS) - ODE": "Xu DR (CS) - ODE",
        "Xu DR (MO) - ODE": "Xu DR (MO) - ODE",
        "Xu DR (FM) - ODE": "Xu DR (FM) - ODE",
        "Xu IPW (CS) - ODE": "Xu IPW (CS) - ODE",
        "Xu IPW (MO) - ODE": "Xu IPW (MO) - ODE",
        "Xu IPW (FM) - ODE": "Xu IPW (FM) - ODE",
    }

    coverage_data = []
    for estimator in target_estimators:
        # Convert to actual estimator name
        actual_name = actual_estimator_names.get(estimator, estimator)

        # Search for corresponding estimator in evaluation results
        matching_rows = evaluation_results[
            evaluation_results["Estimator"].str.contains(
                actual_name, na=False, regex=False
            )
        ]

        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            coverage_rate = row.get("Coverage_Rate", np.nan)
            coverage_data.append(
                {
                    "Estimator": estimator,
                    "Coverage Rate (95% CI)": (
                        coverage_rate if not pd.isna(coverage_rate) else "N/A"
                    ),
                }
            )
        else:
            coverage_data.append(
                {"Estimator": estimator, "Coverage Rate (95% CI)": "N/A"}
            )

    coverage_df = pd.DataFrame(coverage_data)

    if save_path:
        coverage_df.to_csv(save_path, index=False)

    return coverage_df


def create_influence_function_comparison_plot(
    influence_df: pd.DataFrame,
    save_path: Optional[str] = None,
    language: str = "auto",
) -> None:
    """Create box plot comparing distributions of influence functions

    Args:
        influence_df: DataFrame generated by collect_influence_functions_from_simulation()
        save_path: Save path
        language: Language setting
    """
    import numpy as np

    setup_plot_style((10, 6))

    methods = influence_df["method"].unique()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Box plot (influence function values)
    box_data = [
        influence_df[influence_df["method"] == m]["influence_function"].values
        for m in methods
    ]
    bp = ax.boxplot(box_data, labels=methods, patch_artist=True)

    colors = [
        "lightblue",
        "lightcoral",
        "lightgreen",
        "lightyellow",
        "lightpink",
        "lightgray",
    ]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)

    ax.set_ylabel("Influence Function", fontsize=12)
    ax.set_title("Box Plot of Influence Functions", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_plot(save_path, "Figure")

    # Display statistical summary
    print("\n=== Influence Function Statistics ===")
    for method in methods:
        method_data = influence_df[influence_df["method"] == method]
        influence_values = method_data["influence_function"].values
        abs_influence_values = method_data["abs_influence_function"].values
        print(f"\n{method}:")
        print(f"  Mean: {np.mean(influence_values):.6f}")
        print(f"  Median: {np.median(influence_values):.6f}")
        print(f"  Std: {np.std(influence_values):.6f}")
        print(f"  Variance: {np.var(influence_values):.6f}")
        print(f"  Min: {np.min(influence_values):.6f}")
        print(f"  Max: {np.max(influence_values):.6f}")
        print(f"  1st percentile: {np.percentile(influence_values, 1):.6f}")
        print(f"  99th percentile: {np.percentile(influence_values, 99):.6f}")
        print(f"  Mean |Influence|: {np.mean(abs_influence_values):.6f}")
        print(f"  Median |Influence|: {np.median(abs_influence_values):.6f}")


def create_se_comparison_table(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create comparison table of HAC standard errors and bootstrap standard errors

    Args:
        results_df: DataFrame of simulation results (includes HAC SE and Bootstrap SE)
        save_path: Save path (if None, do not save)

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []

    # Get HAC standard error columns (*_se format) and bootstrap standard error columns (*_se_bootstrap format)
    hac_se_cols = [
        col
        for col in results_df.columns
        if col.endswith("_se") and not col.endswith("_se_bootstrap")
    ]
    bootstrap_se_cols = [
        col for col in results_df.columns if col.endswith("_se_bootstrap")
    ]

    # Extract estimator names (e.g., "proposed_adtt_se" -> "proposed_adtt")
    estimator_names = set()
    for col in hac_se_cols:
        estimator_name = col.replace("_se", "")
        estimator_names.add(estimator_name)
    for col in bootstrap_se_cols:
        estimator_name = col.replace("_se_bootstrap", "")
        estimator_names.add(estimator_name)

    # Compare for each estimator
    for estimator_name in sorted(estimator_names):
        hac_se_col = f"{estimator_name}_se"
        bootstrap_se_col = f"{estimator_name}_se_bootstrap"

        if hac_se_col in results_df.columns and bootstrap_se_col in results_df.columns:
            # Calculate mean (excluding NaN)
            hac_se_mean = results_df[hac_se_col].mean()
            bootstrap_se_mean = results_df[bootstrap_se_col].mean()

            # Calculate standard deviation
            hac_se_std = results_df[hac_se_col].std()
            bootstrap_se_std = results_df[bootstrap_se_col].std()

            # Calculate ratio
            ratio = bootstrap_se_mean / hac_se_mean if hac_se_mean != 0 else np.nan

            comparison_data.append(
                {
                    "Estimator": estimator_name,
                    "HAC SE (Mean)": f"{hac_se_mean:.6f}",
                    "HAC SE (Std)": f"{hac_se_std:.6f}",
                    "Bootstrap SE (Mean)": f"{bootstrap_se_mean:.6f}",
                    "Bootstrap SE (Std)": f"{bootstrap_se_std:.6f}",
                    "Ratio (Bootstrap/HAC)": (
                        f"{ratio:.4f}" if not np.isnan(ratio) else "N/A"
                    ),
                }
            )
        elif hac_se_col in results_df.columns:
            hac_se_mean = results_df[hac_se_col].mean()
            hac_se_std = results_df[hac_se_col].std()
            comparison_data.append(
                {
                    "Estimator": estimator_name,
                    "HAC SE (Mean)": f"{hac_se_mean:.6f}",
                    "HAC SE (Std)": f"{hac_se_std:.6f}",
                    "Bootstrap SE (Mean)": "N/A",
                    "Bootstrap SE (Std)": "N/A",
                    "Ratio (Bootstrap/HAC)": "N/A",
                }
            )
        elif bootstrap_se_col in results_df.columns:
            bootstrap_se_mean = results_df[bootstrap_se_col].mean()
            bootstrap_se_std = results_df[bootstrap_se_col].std()
            comparison_data.append(
                {
                    "Estimator": estimator_name,
                    "HAC SE (Mean)": "N/A",
                    "HAC SE (Std)": "N/A",
                    "Bootstrap SE (Mean)": f"{bootstrap_se_mean:.6f}",
                    "Bootstrap SE (Std)": f"{bootstrap_se_std:.6f}",
                    "Ratio (Bootstrap/HAC)": "N/A",
                }
            )

    comparison_df = pd.DataFrame(comparison_data)

    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"Saved standard error comparison table: {save_path}")

    return comparison_df


def create_bootstrap_se_distribution_plot(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Create distribution plot of bootstrap standard errors

    Args:
        results_df: DataFrame of simulation results (includes Bootstrap SE)
        save_path: Save path
    """
    # Get bootstrap standard error columns
    bootstrap_se_cols = [
        col for col in results_df.columns if col.endswith("_se_bootstrap")
    ]

    if not bootstrap_se_cols:
        print("Warning: Bootstrap standard error columns not found.")
        return

    setup_plot_style((12, 8))

    # Extract estimator names
    estimator_names = [col.replace("_se_bootstrap", "") for col in bootstrap_se_cols]

    # Prepare data
    plot_data = []
    for col, name in zip(bootstrap_se_cols, estimator_names):
        values = results_df[col].dropna().values
        if len(values) > 0:
            plot_data.append(values)
        else:
            plot_data.append([])

    # Plot only estimators with valid data
    valid_data = [
        (name, data) for name, data in zip(estimator_names, plot_data) if len(data) > 0
    ]

    if not valid_data:
        print("Warning: No plottable data available.")
        return

    estimator_names_valid = [name for name, _ in valid_data]
    plot_data_valid = [data for _, data in valid_data]

    # Create box plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    bp = ax.boxplot(plot_data_valid, labels=estimator_names_valid, patch_artist=True)

    # Set colors
    colors = [
        "lightblue",
        "lightcoral",
        "lightgreen",
        "lightyellow",
        "lightpink",
        "lightgray",
        "lightcyan",
        "lavender",
    ]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)

    ax.set_ylabel("Bootstrap Standard Error", fontsize=12)
    ax.set_xlabel("Estimator", fontsize=12)
    ax.set_title(
        "Distribution of Bootstrap Standard Errors", fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_plot(save_path, "Figure")

    # Display statistical summary
    print("\n=== Bootstrap Standard Error Statistics ===")
    for name, data in valid_data:
        print(f"\n{name}:")
        print(f"  Mean: {np.mean(data):.6f}")
        print(f"  Median: {np.median(data):.6f}")
        print(f"  Std: {np.std(data):.6f}")
        print(f"  Min: {np.min(data):.6f}")
        print(f"  Max: {np.max(data):.6f}")
        print(f"  25th percentile: {np.percentile(data, 25):.6f}")
        print(f"  75th percentile: {np.percentile(data, 75):.6f}")
