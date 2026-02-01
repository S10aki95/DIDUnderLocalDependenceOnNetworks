import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd


def run_robustness_experiments(
    experiment_types: List[str] = None,
    base_config: str = "default",
    skip_basic_plots: bool = False,
    analyze_influence: bool = False,
    use_bootstrap_se: bool = False,
) -> Dict[str, Any]:
    """Execute robustness check experiments"""
    from src.settings import get_robustness_scenarios

    # Get robustness settings
    scenarios = get_robustness_scenarios()

    if experiment_types is None:
        experiment_types = list(scenarios.keys())

    print(f"\n{'='*80}\nStarting Robustness Check\n{'='*80}")

    all_results = {}
    start_time = datetime.now()

    for exp_type in experiment_types:
        if exp_type not in scenarios:
            print(f"Warning: Unknown experiment type '{exp_type}' will be skipped")
            continue

        scenario = scenarios[exp_type]
        parameter = scenario["parameter"]
        values = scenario["values"]

        print(f"\n--- Running Scenario: {exp_type} ({scenario['display_name']}) ---")
        print(f"Parameter: {parameter}")
        print(f"Description: {scenario['description']}")

        for value in values:
            # Dynamically generate override settings
            overrides = {parameter: value}
            if use_bootstrap_se:
                overrides["use_bootstrap_se"] = True
            # Generate experiment identifier (e.g., sample_size_200)
            experiment_id = f"{exp_type}_{str(value).replace('.', '_')}"

            try:
                # Execute with base config name and override settings
                result = run_robustness_experiment(
                    base_config,
                    overrides=overrides,
                    experiment_id=experiment_id,
                    skip_basic_plots=skip_basic_plots,
                    analyze_influence=analyze_influence,
                )
                all_results[experiment_id] = result
                print(f"✓ Experiment completed: {experiment_id}")
            except Exception as e:
                print(f"✗ Experiment failed: {experiment_id} - {str(e)}")
                all_results[experiment_id] = {"error": str(e)}

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*80}")
    print(f"Robustness Check Completed")
    print(f"Execution time: {duration}")
    print(
        f"Success: {sum(1 for r in all_results.values() if 'error' not in r)}/{len(all_results)}"
    )
    print(f"{'='*80}")

    # Generate robustness analysis charts
    from src.visualization import create_robustness_plots

    print("\nGenerating robustness analysis charts...")
    try:
        create_robustness_plots(all_results, output_dir="results")
        print("✓ Generated robustness analysis charts")
    except Exception as e:
        print(f"⚠ Error occurred while generating charts: {e}")
        import traceback

        traceback.print_exc()

    return all_results


def run_robustness_experiment(
    config_name: str,
    overrides: Optional[Dict[str, Any]] = None,
    experiment_id: str = None,
    skip_basic_plots: bool = False,
    analyze_influence: bool = False,
) -> Dict[str, Any]:
    """Execute robustness experiment"""
    from src.run.simulation import run_simulation_from_config
    from src.settings import get_config

    # Determine display name
    display_name = experiment_id if experiment_id else config_name
    print(f"\n{'='*60}")
    print(f"Running Experiment: {display_name}")
    print(f"{'='*60}")

    # Execute simulation (pass config_name and overrides)
    results_df, evaluation_results, sensitivity_table, true_params, output_dir = (
        run_simulation_from_config(
            config_name, overrides, skip_basic_plots, experiment_id, analyze_influence
        )
    )

    # Get config for report generation
    config = get_config(config_name, overrides)

    return {
        "config_name": config_name,
        "experiment_id": experiment_id or config_name,
        "overrides": overrides,
        "results_df": results_df,
        "evaluation_results": evaluation_results,
        "sensitivity_table": sensitivity_table,
        "true_params": true_params,
        "output_dir": output_dir,
        "config": config,
    }


def main():
    """Main execution function"""
    from src.settings import print_robustness_summary, get_robustness_scenarios

    parser = argparse.ArgumentParser(
        description="SpillOver DID Simulation and Real Data Experiments"
    )

    # Experiment type selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "real_data"],
        default="simulation",
        help="Execution mode: simulation or real_data",
    )

    # Configuration name (optional)
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name (default)",
    )

    # Options for real data experiments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Real data directory (when in real_data mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--use_bootstrap",
        action="store_true",
        help="Enable bootstrap standard error calculation (common to simulation/real_data modes)",
    )
    parser.add_argument(
        "--analyze_influence",
        action="store_true",
        help="Enable influence function distribution analysis (simulation mode)",
    )

    args = parser.parse_args()

    # Execute according to mode
    if args.mode == "simulation":
        run_simulation_mode(args)
    elif args.mode == "real_data":
        run_real_data_mode(args)
    else:
        print(f"Error: Unknown mode '{args.mode}' specified")
        return


def run_simulation_mode(args):
    """Execute simulation mode"""
    from src.settings import print_robustness_summary, get_robustness_scenarios

    print("=" * 80)
    print("SpillOver DID Simulation Experiment - All Charts Generation")
    print("=" * 80)

    # Display robustness settings
    print("\n[Configuration Information]")
    print_robustness_summary()

    # 1. Execute basic experiment (generates Figures 1-3)
    """
    ======================================================================================
    Basic Experiment Processing Flow:
    
    1. Prepare simulation settings    
    2. Execute simulation (run_simulation_from_config)
       - Run M simulations (default: 100 iterations)
       - Calculate the following estimators in each simulation:
         * Proposed methods: Proposed ADTT/AITT (Logistic)
         * Standard methods: Canonical IPW, TWFE, DR-DID, Modified TWFE
         * Xu estimators: DR/IPW (CS/MO/FM exposure types)
       - Calculate HAC standard errors (when spatial coordinate information is available)    
    3. Evaluate and save results
       - Calculate Bias, RMSE, Coverage Rate for each estimator    
    4. Generate visualizations (when skip_basic_plots=False)
       - Figure 1: spillover_structure.png (spillover structure)
       - Figure 2: estimator_distribution_adtt.png (estimator distribution)
       - Figure 3: units_scatter_plot.png (unit placement scatter plot)
       - Table 2: coverage_rate.csv (confidence interval coverage rate)
    5. Integrate results
       - Store basic experiment results in basic_result dictionary
       - Prepare for subsequent robustness experiments
    ======================================================================================
    """
    print("\n[Step 1] Running basic experiment...")
    from src.run.simulation import run_simulation_from_config
    from src.settings import get_config

    # Execute basic simulation (using default settings)
    overrides = {}
    if args.use_bootstrap:
        overrides["use_bootstrap_se"] = True
    results_df, evaluation_results, sensitivity_table, true_params, output_dir = (
        run_simulation_from_config(
            args.config, overrides, analyze_influence=args.analyze_influence
        )
    )
    config = get_config(args.config, overrides)

    basic_result = {
        "config_name": args.config,
        "results_df": results_df,
        "evaluation_results": evaluation_results,
        "sensitivity_table": sensitivity_table,
        "true_params": true_params,
        "output_dir": output_dir,
        "config": config,
    }
    print("✓ Basic experiment completed")

    # 2. Execute robustness experiments (Figures 1-3 are not overwritten)
    """
    ======================================================================================
    Robustness Experiment Processing Flow:
    
    1. Prepare experiment scenarios
       - Get each experiment type: sample size, network density, correlation, feature count
    2. Execute simulation for each scenario
    3. Evaluate and save results
    4. Generate robustness charts
       - Figures 4-5: Sample size vs Bias/RMSE
       - Figures 6-7: Network density vs Bias/RMSE  
       - Figures 8-9: Correlation vs Bias/RMSE
       - Figures 10-11: Feature count vs Bias/RMSE (sensitivity analysis)
    5. Generate integrated CSV file
       - Integrate all robustness experiment results into robustness_results.csv
       - Include experiment type, parameter values, estimator performance metrics
    ======================================================================================
    """
    print("\n[Step 2] Running robustness experiments...")
    scenarios = get_robustness_scenarios()
    all_robustness_types = list(scenarios.keys())

    # Execute robustness experiments (basic charts are not overwritten)
    all_results = run_robustness_experiments(
        all_robustness_types,
        base_config="robustness",
        skip_basic_plots=True,
        analyze_influence=args.analyze_influence,
        use_bootstrap_se=args.use_bootstrap,
    )
    print("✓ Robustness experiments completed")

    # Sensitivity analysis is not executed, so create empty DataFrame
    import pandas as pd

    sensitivity_table = pd.DataFrame()

    # 3. Generate integrated CSV file
    print("\n[Step 3] Generating integrated CSV file...")
    import pandas as pd

    # Integrate all robustness experiment results
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
                    "config_name": result["config_name"],
                    "overrides": (
                        str(result["overrides"]) if result["overrides"] else None
                    ),
                }
            )

    if robustness_data:
        robustness_df = pd.DataFrame(robustness_data)
        robustness_csv_path = "results/robustness_results.csv"
        robustness_df.to_csv(robustness_csv_path, index=False)
        print(f"✓ Generated integrated CSV file: {robustness_csv_path}")

    # 4. Generate integrated report (basic simulation + robustness experiments)
    print("\n[Step 4] Generating integrated report...")
    from src.visualization import generate_results_markdown

    markdown_file = generate_results_markdown(
        basic_result["results_df"],
        basic_result["evaluation_results"],
        sensitivity_table,
        basic_result["true_params"],
        "results",
        basic_result["config"],
        robustness_results=all_results,
        filename="robustness_summary.md",
    )
    print(f"✓ Generated integrated report: {markdown_file}")

    print("\n" + "=" * 80)
    print(
        "All experiments completed - Figures 1-11, Tables 1-2, robustness_results.csv have been generated"
    )
    print("=" * 80)


def run_real_data_mode(args):
    """Execute real data experiment mode"""
    from src.settings import Config

    print("=" * 80)
    print("SEZ Analysis: Evaluation of China's SEZ Policy Effects")
    print("=" * 80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")
    print(f"Bootstrap iterations: {Config().n_bootstrap}")
    print(f"Bootstrap enabled: {args.use_bootstrap}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Execute real data experiment
        from src.settings import load_sez_data
        from src.run.real_data import SEZEstimator
        from src.model.bootstrap import ClusterBootstrap

        # Receive in same output format as dgp.py
        df, neighbors_list, true_params = run_sez_analysis_internal(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_bootstrap_se=args.use_bootstrap,
        )

        print("\n" + "=" * 80)
        print("SEZ Analysis Completed")
        print("=" * 80)
        print(f"Analysis data shape: {df.shape}")
        print(f"Result files:")
        print(f"  - {os.path.join(args.output_dir, 'sez_analysis_results.csv')}")
        print(f"  - {os.path.join(args.output_dir, 'sez_analysis_report.md')}")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()


def run_sez_analysis_internal(
    data_dir: str = "data",
    output_dir: str = "results",
    use_bootstrap_se: bool = False,
) -> tuple:
    """
    Execute SEZ analysis (internal function)

    Args:
        data_dir: Data directory
        output_dir: Output directory
        use_bootstrap_se: Whether to use bootstrap standard errors

    Returns:
        tuple: (DataFrame, List[List[int]], Dict[str, float])
            - df: Analysis dataframe
            - neighbors_list: List of neighbor indices for each unit (empty list for real data)
            - true_params: Dictionary of true parameter values (empty dictionary for real data)
    """
    from src.settings import load_sez_data
    from src.run.real_data import SEZEstimator
    from src.model.bootstrap import ClusterBootstrap
    from src.settings import Config

    n_bootstrap = Config().n_bootstrap

    print("=" * 80)
    print("SEZ Analysis: Evaluation of China's SEZ Policy Effects")
    print("=" * 80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {data_dir}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Bootstrap enabled: {use_bootstrap_se}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    print("\n[Step 1] Loading data...")
    loader = load_sez_data(data_dir)

    # Display data summary
    summary = loader.get_data_summary()
    print("\nData Summary:")
    for name, info in summary.items():
        print(
            f"  {name}: {info['shape']} (Years: {info['years']}, Villages: {info['villages']}, Counties: {info['counties']})"
        )

    # 2-6. Execute analysis for each of the 3 outcome variables
    outcome_vars = {"k": "Capital (log)", "l": "Employment (log)", "y": "Output (log)"}

    all_results = {}  # Store results for all outcome variables
    all_missing_info = []  # Store missing data information for all outcome variables

    for outcome_var, outcome_name in outcome_vars.items():
        print("\n" + "=" * 80)
        print(f"Outcome Variable: {outcome_name} ({outcome_var})")
        print("=" * 80)

        # 2. Prepare analysis data
        print("\n[Step 2] Preparing analysis data...")
        df_analysis, missing_info = loader.create_panel_data(
            outcome_var=outcome_var, return_missing_info=True
        )
        missing_info["outcome_var"] = outcome_var
        missing_info["outcome_name"] = outcome_name

        # Create exposure mapping
        df_with_exposure = loader.create_exposure_mapping(df_analysis)
        print(f"✓ Panel data creation completed: {df_with_exposure.shape}")

        # 3. Prepare covariates
        print("\n[Step 3] Preparing covariates...")
        df_final, PS_covariates = loader.prepare_covariates(df_with_exposure)
        print(f"✓ Covariate preparation completed: {df_final.shape}")

        # 4. Execute estimators
        print("\n[Step 4] Executing doubly robust estimators...")
        # Config for real data analysis (optimized settings applied by default)
        config = Config(use_bootstrap_se=use_bootstrap_se)
        estimator = SEZEstimator(config=config)

        # Define covariates (use those obtained from prepare_covariates)
        Z_list = ["airport_p", "port_p", "kl_p", "num_p"]

        # Estimate effects
        results = estimator.estimate_all_effects(df_final, PS_covariates, Z_list)

        print("\nEstimation Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # 5. Estimate bootstrap standard errors
        print("\n[Step 5] Estimating bootstrap standard errors...")
        se_results = {}
        if estimator.config.use_bootstrap_se:
            bootstrap = ClusterBootstrap(config=estimator.config)
            se_results = bootstrap.bootstrap_standard_errors(
                df_final, estimator, PS_covariates, Z_list
            )
        else:
            print(
                "Skipping bootstrap standard error calculation (use_bootstrap_se=False)"
            )

        # Integrate results
        final_results = {}
        for key, value in results.items():
            final_results[key] = value
            se_key = f"{key}_se"
            if se_key in se_results:
                final_results[se_key] = se_results[se_key]

        print("\nFinal Results (with standard errors):")
        for key, value in final_results.items():
            if key.endswith("_se"):
                print(f"  {key}: {value:.4f}")
            else:
                se_key = f"{key}_se"
                se_value = final_results.get(se_key, "N/A")
                print(f"  {key}: {value:.4f} (SE: {se_value})")

        # Save results (by outcome variable)
        all_results[outcome_var] = {
            "outcome_name": outcome_name,
            "results": final_results,
            "data_shape": df_final.shape,
            "n_treated": df_final["W"].sum(),
            "n_exposed": df_final["G"].sum(),
        }

        # Store missing data information
        all_missing_info.append(missing_info)

    # 6. Save results
    print("\n" + "=" * 80)
    print("[Step 6] Saving results...")
    print("=" * 80)

    # Save results to CSV (by each outcome variable)
    all_results_df = []
    for outcome_var, outcome_data in all_results.items():
        row = {"outcome_var": outcome_var, "outcome_name": outcome_data["outcome_name"]}
        row.update(outcome_data["results"])
        all_results_df.append(row)

    results_df = pd.DataFrame(all_results_df)
    results_csv_path = os.path.join(output_dir, "sez_analysis_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Saved results: {results_csv_path}")

    # Save detailed results to Markdown file
    markdown_path = os.path.join(output_dir, "sez_analysis_report.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("# SEZ Analysis Results Report\n\n")
        f.write(
            f"**Execution Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write(
            "This report presents analysis results for 3 outcome variables based on Xu (2025) paper.\n\n"
        )

        # Output results for each outcome variable
        for outcome_var, outcome_data in all_results.items():
            outcome_name = outcome_data["outcome_name"]
            final_results = outcome_data["results"]

            f.write(f"## Outcome Variable: {outcome_name}\n\n")

            f.write("### Data Summary\n\n")
            f.write(f"- Analysis data shape: {outcome_data['data_shape']}\n")
            f.write(
                f"- Number of SEZ-designated villages: {outcome_data['n_treated']}\n"
            )
            f.write(
                f"- Number of highly exposed villages: {outcome_data['n_exposed']}\n\n"
            )

            f.write("### Estimation Results\n\n")
            f.write("| Effect | Estimate | Standard Error |\n")
            f.write("|--------|----------|----------------|\n")

            for key, value in final_results.items():
                if not key.endswith("_se"):
                    se_key = f"{key}_se"
                    se_value = final_results.get(se_key, "N/A")
                    if isinstance(se_value, (int, float)):
                        f.write(f"| {key} | {value:.4f} | {se_value:.4f} |\n")
                    else:
                        f.write(f"| {key} | {value:.4f} | {se_value} |\n")

            f.write("\n")

        f.write("## Effect Descriptions\n\n")
        f.write("### Xu (2025) Methods (Considering Spillover Effects)\n\n")
        f.write("- **DATT_0**: Direct effect in weakly exposed villages\n")
        f.write("- **DATT_1**: Direct effect in strongly exposed villages\n")
        f.write("- **ODE**: Overall direct effect (weighted average)\n")
        f.write("- **Spillover_Effect**: Spillover effect\n")
        f.write(
            "- **Spillover_Treated_DR/IPW**: Spillover effect in treatment group (W=1) τ(1,1,0)\n"
        )
        f.write(
            "- **Spillover_Control_DR/IPW**: Spillover effect in control group (W=0) τ(0,1,0)\n\n"
        )
        f.write("### Canonical DID (Standard DID, Ignoring Spillover Effects)\n\n")
        f.write("- **Canonical_IPW**: Standard IPW-DID estimator (Abadie, 2005)\n")
        f.write("- **Canonical_DR_DID**: Standard DR-DID estimator\n")
        f.write("- Used to demonstrate bias when spillover effects are ignored\n")
        f.write("- Used for comparison with Table 4 of the paper\n\n")

        f.write("## Notes\n\n")
        f.write(
            "- Standard errors are estimated using cluster bootstrap method (county-level clustering)\n"
        )
        f.write(
            "- Exposure mapping is based on Leave-one-out SEZ ratio within counties\n"
        )
        f.write(
            "- Covariates include village characteristics and county-level mean values\n"
        )
        f.write(f"- Bootstrap iterations: {n_bootstrap}\n")

    print(f"✓ Saved detailed report: {markdown_path}")

    # Save missing data analysis
    missing_data_df = pd.DataFrame(all_missing_info)
    # Add Xu (2025) paper sample sizes (these need to be manually set based on the paper)
    # Note: These values should be updated based on actual paper values
    xu_paper_n_values = {
        "k": None,  # Set actual value from paper if available
        "l": None,  # Set actual value from paper if available
        "y": None,  # Set actual value from paper if available
    }
    missing_data_df["xu_paper_n"] = missing_data_df["outcome_var"].map(
        xu_paper_n_values
    )
    missing_data_df["difference"] = missing_data_df.apply(
        lambda row: (
            row["n_after_dropna"] - row["xu_paper_n"]
            if row["xu_paper_n"] is not None
            else None
        ),
        axis=1,
    )

    # Reorder columns
    column_order = [
        "outcome_var",
        "outcome_name",
        "n_villages_2004",
        "n_villages_2008",
        "n_villages_both",
        "n_after_merge",
        "n_complete",
        "n_after_dropna",
        "xu_paper_n",
        "difference",
    ]
    missing_data_df = missing_data_df[column_order]

    missing_data_csv_path = os.path.join(output_dir, "missing_data_analysis.csv")
    missing_data_df.to_csv(missing_data_csv_path, index=False)
    print(f"✓ Saved missing data analysis: {missing_data_csv_path}")

    # Return dataframe of last processed outcome variable (for backward compatibility)
    df_final = loader.create_panel_data(outcome_var="y")
    df_with_exposure = loader.create_exposure_mapping(df_final)
    df_final, _ = loader.prepare_covariates(df_with_exposure)

    # Return in same output format as dgp.py
    # For real data, neighbor list is empty, true parameters are also empty
    neighbors_list = []  # Neighbor information is not used in real data
    true_params = {}  # True parameters are unknown in real data

    return df_final, neighbors_list, true_params


if __name__ == "__main__":
    main()
