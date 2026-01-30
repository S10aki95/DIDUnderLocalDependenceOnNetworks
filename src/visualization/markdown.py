"""
Markdown report generation module

Generates Markdown reports of simulation results.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

from .helpers import (
    classify_experiments_by_type,
    map_estimator_names,
    _generate_estimator_descriptions,
)


def _generate_header() -> List[str]:
    """Generate Markdown report header"""
    return [
        "# SpillOver DID Simulation Results Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]


def _generate_true_params_section(true_params: Dict[str, float]) -> List[str]:
    """Generate true parameter values section"""
    return [
        "## True Parameter Values",
        "",
        "The following values are true parameter values theoretically calculated from the data generation process (DGP).",
        "We evaluate estimator performance by comparing these values with estimates from each estimator.",
        "",
        f"- **ADTT (Average Direct Treatment Effect on Treated)**: {true_params['adtt']:.4f}",
        "  - Average direct effect on treated units, given the treatment status of neighbors",
        "  - Theoretically equal to direct effect τ",
        "",
        f"- **AITT (Average Indirect Treatment Effect on Treated)**: {true_params['aitt']:.4f}",
        "  - Average effect of treatment received by treated units on their neighboring units",
        "  - Calculated as average of stepwise changes in spillover effects",
        "",
    ]


def _generate_simulation_config_section(
    results_df: pd.DataFrame, config: Dict[str, Any]
) -> List[str]:
    """Generate simulation configuration section"""
    content = [
        "## Simulation Configuration",
        "",
        "This simulation was executed based on the experimental settings in README Section 3.1.",
        "It uses a data generation process (DGP) with complex spillover structure and",
        "evaluates estimator performance under conditions where exposure mapping is difficult to identify.",
        "",
        "### Basic Settings",
        "",
        f"- **Number of simulations**: {len(results_df)}",
    ]

    # Get number of units
    n_units = None
    if "n_units" in results_df.columns:
        n_units = results_df["n_units"].iloc[0]
    else:
        n_units = config.n_units

    content.extend(
        [
            f"- **Number of units**: {n_units}",
            f"- **Interference range distance (K)**: {config.k_distance}",
            f"- **Space size**: {config.space_size} × {config.space_size}",
            f"- **Random seed**: {config.random_seed} (fixed for reproducibility)",
            "",
        ]
    )

    return content


def _generate_visualization_section(
    output_dir: str, config: Dict[str, Any]
) -> List[str]:
    """Generate visualization results section"""
    content = ["## Visualization Results", ""]

    # Spillover structure plot
    spillover_plot_path = os.path.join(output_dir, "spillover_structure.png")
    if os.path.exists(spillover_plot_path):
        k_distance = config.k_distance
        content.extend(
            [
                "### Figure 1: Distribution of Spillover Effect Counts",
                "",
                "This figure shows the distribution of neighbor treatment counts (S_i) for each unit generated in the simulation.",
                f"It visualizes the frequency distribution of the number of neighboring treated units (within distance K={k_distance}) for each unit.",
                "",
                f"![Spillover Structure](./spillover_structure.png)",
                "",
                f"*Figure 1: Distribution of spillover effect counts. Shows the distribution of neighbor treatment counts for each unit.*",
                "",
            ]
        )

    # Estimator distribution plot
    distribution_plot_path = os.path.join(output_dir, "estimator_distribution_adtt.png")
    if os.path.exists(distribution_plot_path):
        content.extend(
            [
                "### Figure 2: Estimator Distribution",
                "",
                "This figure shows the distribution of each estimator using box plots.",
                "By comparing with true parameter values (red horizontal line), bias and variance of each estimator can be visually confirmed.",
                "",
                f"![Estimator Distribution](./estimator_distribution_adtt.png)",
                "",
                "*Figure 2: Distribution of each estimator. Red horizontal line indicates true parameter value.*",
                "",
            ]
        )

    # Unit placement scatter plot
    units_scatter_path = os.path.join(output_dir, "units_scatter_plot.png")
    if os.path.exists(units_scatter_path):
        space_size = config.space_size
        content.extend(
            [
                "### Figure 3: Unit Placement and Treatment Status",
                "",
                "This figure shows the spatial placement of units generated in the simulation and their treatment assignment status.",
                f"Each unit is randomly placed in 2D space ({space_size} × {space_size}),",
                "and displayed separately for treatment group (red) and control group (blue).",
                "",
                f"![Unit Placement and Treatment Status](./units_scatter_plot.png)",
                "",
                "*Figure 3: Spatial placement of units and treatment assignment. Red points indicate treatment group, blue points indicate control group.*",
                "",
            ]
        )

    return content


def generate_results_markdown(
    results_df: pd.DataFrame,
    evaluation_results: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
    true_params: Dict[str, float],
    output_dir: str,
    config: Dict[str, Any],
    robustness_results: Optional[Dict[str, Any]] = None,
    filename: str = "simulation_results_report.md",
) -> str:
    """
    Function to generate Markdown report of simulation results

    Args:
        results_df: Dataframe of simulation results
        evaluation_results: Dataframe of evaluation results
        sensitivity_table: Dataframe of sensitivity analysis table
        true_params: True parameter values
        output_dir: Output directory
        config: Configuration dictionary (required)

    Returns:
        str: Path to Markdown file
    """
    markdown_content = []

    # Map estimator names
    evaluation_results = map_estimator_names(evaluation_results)

    # Generate each section
    markdown_content.extend(_generate_header())
    markdown_content.extend(_generate_true_params_section(true_params))
    markdown_content.extend(_generate_simulation_config_section(results_df, config))
    markdown_content.extend(_generate_visualization_section(output_dir, config))

    # Get values from configuration
    k_distance = config.k_distance
    space_size = config.space_size

    # Details of data generation process (DGP)
    markdown_content.append("### Details of Data Generation Process (DGP)")
    markdown_content.append("")
    markdown_content.append("#### 1. Unit Placement")
    markdown_content.append(
        f"- Each unit is randomly placed in 2D space $[0, {space_size}] \\times [0, {space_size}]$ with uniform distribution"
    )
    markdown_content.append(
        f"- Neighbor relationship is defined within Chebyshev distance $K = {k_distance}$"
    )
    markdown_content.append(
        "- Number of neighbor units for each unit varies depending on spatial placement"
    )
    markdown_content.append("")

    markdown_content.append("#### 2. Covariate Generation")

    # Get parameters for covariate generation from configuration
    z_u_correlation_base = config.z_u_correlation_base
    markdown_content.append("  - Spatially correlated between neighbor units")
    markdown_content.append(
        f"  - Base parameter for spatial correlation: $\\rho_0 = {z_u_correlation_base}$"
    )
    markdown_content.append("  - Correlation decays with distance")
    markdown_content.append("")

    markdown_content.append("#### 3. Treatment Assignment")

    # Get coefficients for treatment assignment from configuration
    treatment_z_coef = config.treatment_z_coef
    treatment_z_u_coef = config.treatment_z_u_coef

    markdown_content.append(
        f"- Treatment probability: $Pr(D_i=1 \\mid z_i, z_{{u,i}}) = \\text{{logit}}^{{-1}}({treatment_z_coef} z_i + {treatment_z_u_coef} z_{{u,i}})$"
    )
    markdown_content.append(
        "- Depends on both individual attributes and spatial correlation variables"
    )
    markdown_content.append("- Generates spatially clustered treatment patterns")
    markdown_content.append("")

    markdown_content.append("#### 4. Outcome Generation")

    # Get coefficients for outcome generation from configuration
    beta_1 = config.beta_1
    beta_2 = config.beta_2
    delta = config.delta
    tau = config.tau
    gamma_1 = config.gamma_1
    gamma_2 = config.gamma_2
    y1_error_std = config.y1_error_std
    y2_error_std = config.y2_error_std

    markdown_content.append(
        f"- **Time 1**: $Y_{{1i}} = {beta_1} z_i + {beta_2} z_{{u,i}} + \\epsilon_{{1i}}$ where $\\epsilon_{{1i}} \\sim N(0, {y1_error_std})$"
    )
    markdown_content.append(
        f"- **Time 2**: $Y_{{2i}} = {delta} + Y_{{1i}} + {tau} D_i + f(S_i) + {gamma_1} z_{{u,i}} + {gamma_2} z_i + \\epsilon_{{2i}}$ where $\\epsilon_{{2i}} \\sim N(0, {y2_error_std})$"
    )
    markdown_content.append(f"- **Direct effect**: $\\tau = {tau}$")
    markdown_content.append(
        "- **Spillover effect**: $f(S_i)$ is a non-monotonic function of neighbor treatment count $S_i$"
    )
    markdown_content.append("")

    markdown_content.append("#### 5. Characteristics of Spillover Structure")

    # Get spillover effect settings
    spillover_effects = config.spillover_effects

    markdown_content.append(
        "- **Non-monotonicity**: Effect does not increase monotonically even as neighbor treatment count increases"
    )
    markdown_content.append(
        "- **Negative externality from overcrowding**: Too many treated units causes adverse effects"
    )
    markdown_content.append(f"- **Stages of spillover effects**: {spillover_effects}")
    markdown_content.append(
        "- **Exposure mapping**: Functional form of $f(\\cdot)$ is unknown to researchers"
    )
    markdown_content.append("")

    # Evaluation results table
    markdown_content.append("## Evaluation Results")
    markdown_content.append("")

    if not evaluation_results.empty:
        # Get number of simulations
        n_simulations = config.n_simulations

        if n_simulations == 1:
            markdown_content.append(
                "### Table 1: Estimator Estimation Results (Single Simulation)"
            )
            markdown_content.append("")
            markdown_content.append(
                "This table shows the estimates from each estimator obtained in a single simulation."
            )
            markdown_content.append("")
            markdown_content.append(
                "**Note**: Statistical evaluation is limited with a single simulation."
            )
            markdown_content.append(
                "For statistical evaluation with multiple simulations, please refer to robustness check experiments."
            )
            markdown_content.append("")
            markdown_content.extend(_generate_estimator_descriptions())
            markdown_content.append(
                "| Estimator | Estimate | Difference from True Value |"
            )
            markdown_content.append(
                "|----------|----------|----------------------------|"
            )

            for _, row in evaluation_results.iterrows():
                estimator = row["Estimator"]
                bias = row["Bias"]
                true_value = (
                    true_params.get("adtt", 0.0)
                    if "ADTT" in estimator
                    else true_params.get("aitt", 0.0)
                )
                estimated_value = true_value + bias
                markdown_content.append(
                    f"| {estimator} | {estimated_value:.4f} | {bias:+.4f} |"
                )
        else:
            markdown_content.append(
                "### Table 1: Point Estimation Accuracy Evaluation (Bias and RMSE)"
            )
            markdown_content.append("")
            markdown_content.append(
                "This table shows the results of evaluating point estimation accuracy of each estimator based on evaluation metrics in README Section 3.4."
            )
            markdown_content.append("")
            markdown_content.append("**Description of Evaluation Metrics:**")
            markdown_content.append(
                "- **Bias**: Metric indicating how systematically the expected value of the estimator deviates from the true value"
            )
            markdown_content.append(
                "  - Formula: $\\text{Bias}(\\hat{\\tau}) = \\left( \\frac{1}{M}\\sum_{m=1}^M \\hat{\\tau}_m \\right) - \\tau_{true}$"
            )
            markdown_content.append("  - Ideal value: 0 (closer to zero is better)")
            markdown_content.append(
                "- **RMSE (Root Mean Squared Error)**: Metric indicating typical magnitude of estimation error"
            )
            markdown_content.append(
                "  - Formula: $\\text{RMSE}(\\hat{\\tau}) = \\sqrt{\\frac{1}{M}\\sum_{m=1}^M (\\hat{\\tau}_m - \\tau_{true})^2}$"
            )
            markdown_content.append(
                "  - Reflects both bias and variance, smaller values indicate higher accuracy"
            )
            markdown_content.append("")
            markdown_content.extend(_generate_estimator_descriptions())
            markdown_content.append("| Estimator | Bias | RMSE |")
            markdown_content.append("|----------|------|------|")

            for _, row in evaluation_results.iterrows():
                bias = f"{row['Bias']:.4f}" if not pd.isna(row["Bias"]) else "N/A"
                rmse = f"{row['RMSE']:.4f}" if not pd.isna(row["RMSE"]) else "N/A"
                markdown_content.append(f"| {row['Estimator']} | {bias} | {rmse} |")

        markdown_content.append("")

        # Coverage rate table (if available)
        if "Coverage_Rate" in evaluation_results.columns:
            # Display only Proposed IPW (ADTT) and Canonical IPW according to README Section 5.2 requirements
            target_estimators = ["Proposed IPW ADTT (Logistic)", "Canonical IPW"]
            coverage_df = evaluation_results[
                evaluation_results["Estimator"].isin(target_estimators)
            ][["Estimator", "Coverage_Rate"]].dropna()
            if not coverage_df.empty:
                # Get confidence interval coefficient from configuration
                confidence_level = config.confidence_level
                z_critical = config.z_critical

                if n_simulations == 1:
                    markdown_content.append(
                        f"### Table 2: Confidence Interval Construction (Single Simulation)"
                    )
                    markdown_content.append("")
                    markdown_content.append(
                        "This table shows the confidence intervals constructed in a single simulation."
                    )
                    markdown_content.append("")
                    markdown_content.append(
                        "**Note**: Statistical evaluation of coverage rate is not possible with a single simulation."
                    )
                    markdown_content.append(
                        "For statistical evaluation with multiple simulations, please refer to robustness check experiments."
                    )
                    markdown_content.append("")
                    markdown_content.append(
                        "**Confidence Interval Construction Method:**"
                    )
                    markdown_content.append(
                        f"- **Confidence level**: {int(confidence_level*100)}%"
                    )
                    markdown_content.append(
                        f"- **Construction formula**: $CI = \\left[ \\hat{{\\tau}} - {z_critical} \\times \\widehat{{SE}}(\\hat{{\\tau}}), \\quad \\hat{{\\tau}} + {z_critical} \\times \\widehat{{SE}}(\\hat{{\\tau}}) \\right]$"
                    )
                    markdown_content.append("")
                    markdown_content.append(
                        "| Estimator | Estimate | Standard Error | Confidence Interval Lower | Confidence Interval Upper |"
                    )
                    markdown_content.append(
                        "|----------|----------|----------------|--------------------------|--------------------------|"
                    )

                    for _, row in coverage_df.iterrows():
                        estimator = row["Estimator"]
                        bias = evaluation_results[
                            evaluation_results["Estimator"] == estimator
                        ]["Bias"].iloc[0]
                        true_value = (
                            true_params.get("adtt", 0.0)
                            if "ADTT" in estimator
                            else true_params.get("aitt", 0.0)
                        )
                        estimated_value = true_value + bias

                        # Use correct SE estimate (if HAC SE is calculated)
                        se_column = None
                        if "Proposed IPW ADTT" in estimator:
                            se_column = "proposed_adtt_se"
                        elif "Proposed DR ADTT" in estimator:
                            se_column = "proposed_dr_adtt_se"
                        elif "Proposed IPW AITT" in estimator:
                            se_column = "proposed_aitt_se"
                        elif "Proposed DR AITT" in estimator:
                            se_column = "proposed_dr_aitt_se"
                        elif "Canonical IPW" in estimator:
                            se_column = "canonical_ipw_se"

                        if se_column and se_column in results_df.columns:
                            se = (
                                results_df[se_column].iloc[0]
                                if not results_df[se_column].isna().all()
                                else abs(bias)
                            )
                        else:
                            # Fallback: For a single simulation, RMSE = |Bias|
                            se = abs(bias)

                        ci_lower = estimated_value - z_critical * se
                        ci_upper = estimated_value + z_critical * se
                        markdown_content.append(
                            f"| {estimator} | {estimated_value:.4f} | {se:.4f} | {ci_lower:.4f} | {ci_upper:.4f} |"
                        )
                else:
                    markdown_content.append(
                        f"### Table 2: Validity Assessment of Interval Estimation (Coverage Rate @ {int(confidence_level*100)}%)"
                    )
                    markdown_content.append("")
                    markdown_content.append(
                        "This table shows the results of evaluating the validity of interval estimation for each estimator based on the Coverage Rate metric in README Section 3.4."
                    )
                    markdown_content.append("")
                    markdown_content.append("**Coverage Rate Explanation:**")
                    markdown_content.append(
                        f"- **Definition**: The proportion of {int(confidence_level*100)}% confidence intervals constructed in each simulation that correctly contain the true parameter value"
                    )
                    markdown_content.append(
                        "- **Formula**: $\\text{Coverage Rate} = \\frac{1}{M} \\sum_{m=1}^M \\mathbb{I}(\\tau_{true} \\in CI_m)$"
                    )

                    markdown_content.append(
                        f"- **Confidence Interval Construction**: $CI_m = \\left[ \\hat{{\\tau}}_m - {z_critical} \\times \\widehat{{SE}}(\\hat{{\\tau}}_m), \\quad \\hat{{\\tau}}_m + {z_critical} \\times \\widehat{{SE}}(\\hat{{\\tau}}_m) \\right]$"
                    )
                    markdown_content.append(
                        f"- **Ideal Value**: Closer to {confidence_level} ({int(confidence_level*100)}%) indicates that standard error estimation is appropriate"
                    )
                    markdown_content.append("")
                    markdown_content.append("| Estimator | Coverage Rate |")
                    markdown_content.append("|--------|---------------|")

                    for _, row in coverage_df.iterrows():
                        coverage = (
                            f"{row['Coverage_Rate']:.4f}"
                            if not pd.isna(row["Coverage_Rate"])
                            else "N/A"
                        )
                        markdown_content.append(f"| {row['Estimator']} | {coverage} |")

                markdown_content.append("")

    markdown_content.append("### File List")
    markdown_content.append("")
    markdown_content.append("List of files generated in this simulation experiment.")
    markdown_content.append(
        "Result files are organized according to the project structure in README Section 4.1."
    )
    markdown_content.append("")
    markdown_content.append("**Generated Files:**")
    markdown_content.append(
        "- **`simulation_results.csv`**: Detailed results of all simulations"
    )
    markdown_content.append(
        "  - Estimated values for each estimator at each simulation iteration"
    )
    markdown_content.append("  - Used for comparison with true parameter values")
    markdown_content.append(
        "- **`evaluation_results.csv`**: Evaluation metrics for estimators"
    )
    markdown_content.append(
        "  - Calculation results of evaluation metrics (Bias, RMSE, Coverage Rate) from README Section 3.4"
    )
    markdown_content.append(
        "  - Quantitative comparison of each estimator's performance"
    )
    markdown_content.append(
        "- **`spillover_structure.png`**: Spillover structure diagram (Figure 1)"
    )
    markdown_content.append(
        "  - Visualization of complex spillover structure from README Section 3.1.4"
    )
    markdown_content.append(
        "  - Spatial distribution of non-monotonic interference effects"
    )
    markdown_content.append(
        "- **`estimator_distribution_adtt.png`**: Estimator distribution diagram (Figure 2)"
    )
    markdown_content.append("  - Distribution of each estimator displayed as box plots")
    markdown_content.append(
        "  - Visual evaluation through comparison with true parameter values"
    )
    markdown_content.append("")
    markdown_content.append(
        "By referring to these files, more detailed analysis and reproducibility can be ensured."
    )
    markdown_content.append(
        "Additionally, simulations with different settings can be run following the execution method in README Section 4.3."
    )

    # Add robustness experiment results
    if robustness_results:
        markdown_content.append("")
        markdown_content.append("## Robustness Check Experiment Results")
        markdown_content.append("")
        markdown_content.append(
            "The following tables show the results of robustness check experiments with different settings."
        )
        markdown_content.append(
            "For each experiment type, the performance of estimators (Bias, RMSE, Coverage Rate) can be compared."
        )
        markdown_content.append("")

        # Organize results by experiment type
        experiment_groups = classify_experiments_by_type(robustness_results)

        # Map estimator names in robustness experiment results as well
        for exp_type, experiments in experiment_groups.items():
            for experiment_id, result in experiments:
                if "evaluation_results" in result:
                    result["evaluation_results"] = map_estimator_names(
                        result["evaluation_results"]
                    )

        # Display results for each experiment type
        for exp_type, experiments in experiment_groups.items():
            if not experiments:
                continue

            if exp_type == "sample_size":
                markdown_content.append("### Sample Size Experiment")
                # Figure 4: Relationship between sample size and Bias
                bias_plot_path = os.path.join(
                    output_dir, "robustness_bias_vs_sample_size.png"
                )
                if os.path.exists(bias_plot_path):
                    markdown_content.extend(
                        [
                            "",
                            "#### Figure 4: Relationship between Sample Size and Bias",
                            "",
                            "This figure shows how each estimator's Bias changes with changes in sample size.",
                            "",
                            f"![Relationship between Sample Size and Bias](./robustness_bias_vs_sample_size.png)",
                            "",
                            "*Figure 4: Relationship between sample size and Bias. Shows how each estimator's Bias changes as sample size increases.*",
                            "",
                        ]
                    )
                # Figure 5: Relationship between sample size and RMSE
                rmse_plot_path = os.path.join(
                    output_dir, "robustness_rmse_vs_sample_size.png"
                )
                if os.path.exists(rmse_plot_path):
                    markdown_content.extend(
                        [
                            "#### Figure 5: Relationship between Sample Size and RMSE",
                            "",
                            "This figure shows how each estimator's RMSE changes with changes in sample size.",
                            "",
                            f"![Relationship between Sample Size and RMSE](./robustness_rmse_vs_sample_size.png)",
                            "",
                            "*Figure 5: Relationship between sample size and RMSE. Shows how each estimator's RMSE changes as sample size increases.*",
                            "",
                        ]
                    )
            elif exp_type == "density":
                markdown_content.append("### Network Density Experiment")
                # Figure 6: Relationship between network density and Bias
                bias_plot_path = os.path.join(
                    output_dir, "robustness_bias_vs_density.png"
                )
                if os.path.exists(bias_plot_path):
                    markdown_content.extend(
                        [
                            "",
                            "#### Figure 6: Relationship between Network Density and Bias",
                            "",
                            "This figure shows how each estimator's Bias changes with changes in network density (interference range K).",
                            "",
                            f"![Relationship between Network Density and Bias](./robustness_bias_vs_density.png)",
                            "",
                            "*Figure 6: Relationship between network density and Bias. Shows how each estimator's Bias changes as network density changes.*",
                            "",
                        ]
                    )
                # Figure 7: Relationship between network density and RMSE
                rmse_plot_path = os.path.join(
                    output_dir, "robustness_rmse_vs_density.png"
                )
                if os.path.exists(rmse_plot_path):
                    markdown_content.extend(
                        [
                            "#### Figure 7: Relationship between Network Density and RMSE",
                            "",
                            "This figure shows how each estimator's RMSE changes with changes in network density (interference range K).",
                            "",
                            f"![Relationship between Network Density and RMSE](./robustness_rmse_vs_density.png)",
                            "",
                            "*Figure 7: Relationship between network density and RMSE. Shows how each estimator's RMSE changes as network density changes.*",
                            "",
                        ]
                    )
            elif exp_type == "correlation":
                markdown_content.append("### Spatial Correlation Experiment")
                # Figure 8: Relationship between spatial correlation and Bias
                bias_plot_path = os.path.join(
                    output_dir, "robustness_bias_vs_correlation.png"
                )
                if os.path.exists(bias_plot_path):
                    markdown_content.extend(
                        [
                            "",
                            "#### Figure 8: Relationship between Spatial Correlation and Bias",
                            "",
                            "This figure shows how each estimator's Bias changes with changes in the strength of spatial correlation.",
                            "",
                            f"![Relationship between Spatial Correlation and Bias](./robustness_bias_vs_correlation.png)",
                            "",
                            "*Figure 8: Relationship between spatial correlation and Bias. Shows how each estimator's Bias changes as the strength of spatial correlation changes.*",
                            "",
                        ]
                    )
                # Figure 9: Relationship between spatial correlation and RMSE
                rmse_plot_path = os.path.join(
                    output_dir, "robustness_rmse_vs_correlation.png"
                )
                if os.path.exists(rmse_plot_path):
                    markdown_content.extend(
                        [
                            "#### Figure 9: Relationship between Spatial Correlation and RMSE",
                            "",
                            "This figure shows how each estimator's RMSE changes with changes in the strength of spatial correlation.",
                            "",
                            f"![Relationship between Spatial Correlation and RMSE](./robustness_rmse_vs_correlation.png)",
                            "",
                            "*Figure 9: Relationship between spatial correlation and RMSE. Shows how each estimator's RMSE changes as the strength of spatial correlation changes.*",
                            "",
                        ]
                    )
            elif exp_type == "features":
                markdown_content.append("### Neighbor Feature Count Experiment")
                # Figure 10: Relationship between neighbor feature count and Bias (Proposed IPW (ADTT) only)
                bias_plot_path = os.path.join(
                    output_dir, "sensitivity_bias_vs_features.png"
                )
                if os.path.exists(bias_plot_path):
                    markdown_content.extend(
                        [
                            "",
                            "#### Figure 10: Relationship between Neighbor Feature Count and Bias (Proposed IPW (ADTT) only)",
                            "",
                            "This figure shows how the Proposed IPW (ADTT) estimator's Bias changes with changes in neighbor feature count.",
                            "",
                            f"![Relationship between Neighbor Feature Count and Bias](./sensitivity_bias_vs_features.png)",
                            "",
                            "*Figure 10: Relationship between neighbor feature count and Bias. Shows how the Proposed IPW (ADTT) estimator's Bias changes as neighbor feature count changes.*",
                            "",
                        ]
                    )
                # Figure 11: Relationship between neighbor feature count and RMSE (Proposed IPW (ADTT) only)
                rmse_plot_path = os.path.join(
                    output_dir, "sensitivity_rmse_vs_features.png"
                )
                if os.path.exists(rmse_plot_path):
                    markdown_content.extend(
                        [
                            "#### Figure 11: Relationship between Neighbor Feature Count and RMSE (Proposed IPW (ADTT) only)",
                            "",
                            "This figure shows how the Proposed IPW (ADTT) estimator's RMSE changes with changes in neighbor feature count.",
                            "",
                            f"![Relationship between Neighbor Feature Count and RMSE](./sensitivity_rmse_vs_features.png)",
                            "",
                            "*Figure 11: Relationship between neighbor feature count and RMSE. Shows how the Proposed IPW (ADTT) estimator's RMSE changes as neighbor feature count changes.*",
                            "",
                        ]
                    )

    # Save Markdown file
    markdown_file = os.path.join(output_dir, filename)
    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_content))

    return markdown_file
