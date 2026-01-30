"""
Simulation execution module

Executes simulations based on requirements in README Chapter 3 and evaluates and visualizes results.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
import multiprocessing

from ..settings import get_config, Config
from ..settings import generate_data
from ..model.standard import (
    estimate_twfe,
    estimate_modified_twfe,
)
from ..utils import (
    evaluate_estimators,
)
from ..visualization import (
    create_spillover_plot,
    create_distribution_plot_adtt,
    create_distribution_plot_aitt,
    create_units_scatter_plot,
    create_coverage_rate_table,
    create_robustness_plots,
    cleanup_old_results,
    collect_influence_functions_from_simulation,
    create_influence_function_comparison_plot,
    create_se_comparison_table,
    create_bootstrap_se_distribution_plot,
)
from ..model.bootstrap import StandardBootstrap
from .common import (
    compute_all_estimators,
    _generate_iteration_seeds,
    _print_simulation_config,
    _generate_experiment_id,
    ESTIMATOR_COLUMNS,
    SE_COLUMNS,
)


def run_single_simulation(
    n_units: int,
    K: float,
    space_size: float,
    dgp_config: Config,
    random_seed: Optional[int] = None,
    progress_bar: Optional[tqdm] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Function to execute a single simulation

    Calculates each estimator based on experimental settings in README Section 3.1.

    Args:
        n_units: Number of units (README Section 3.1: N = 500)
        K: Neighbor distance (README Section 3.1: K = 1.0)
        space_size: Space size (README Section 3.1: 20.0 × 20.0)
        dgp_config: DGP configuration object (with overrides applied)
        random_seed: Random seed

    Returns:
        Tuple[Dict, pd.DataFrame, List[List[int]]]: (Dictionary of simulation results, generated dataframe, neighbor list)
    """
    # Generate data (DGP from README Section 3.1)
    df, neighbors_list, true_params = generate_data(
        n_units=n_units,
        K=K,
        space_size=space_size,
        dgp_config=dgp_config,
        random_seed=random_seed,
    )

    # Get unit coordinates (for HAC standard error)
    locations = np.column_stack([df["x"], df["y"]]) if "x" in df.columns else None

    # Calculate all estimators using common function
    results = compute_all_estimators(
        df=df,
        neighbors_list=neighbors_list,
        config=dgp_config,
        locations=locations,
        K=K,
        covariates=["z"],
        treatment_col="D",
        compute_standard_se=True,
        random_seed=random_seed,
    )

    # TWFE and Modified TWFE are calculated separately (not included in common function)
    twfe_result = estimate_twfe(df)
    results["canonical_twfe"] = twfe_result.estimate
    results["canonical_twfe_se"] = twfe_result.standard_error

    mod_twfe_result = estimate_modified_twfe(df, neighbors_list)
    results["modified_twfe"] = mod_twfe_result.estimate
    results["modified_twfe_se"] = mod_twfe_result.standard_error

    # Calculate bootstrap standard errors
    if dgp_config.use_bootstrap_se:
        try:
            bootstrap = StandardBootstrap(config=dgp_config)
            # Pass progress bar to display progress
            bootstrap_se_results = bootstrap.bootstrap_standard_errors(
                df=df,
                neighbors_list=neighbors_list,
                config=dgp_config,
                locations=locations,
                K=K,
                covariates=["z"],
                treatment_col="D",
                progress_bar=progress_bar,
            )
            # Add bootstrap standard errors to results
            results.update(bootstrap_se_results)
        except Exception as e:
            if dgp_config.verbose:
                print(f"Warning: Bootstrap standard error calculation failed: {e}")

    # Add true parameter values
    results.update(true_params)

    # Add number of units
    results["n_units"] = n_units

    return results, df, neighbors_list


def run_simulation_experiment(
    config_name: str = "default",
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Function to execute simulation experiment

    Executes M simulations based on experimental settings in README Section 3.1.

    Args:
        config_name: Base configuration name
        overrides: Dictionary of settings to override (supports dot notation)

    Returns:
        Tuple[pd.DataFrame, Dict, List[pd.DataFrame], List[List[List[int]]]]:
        (Results dataframe, true parameter values, list of dataframes, list of neighbor lists)
    """
    print("=" * 60)
    print("SpillOver DID Simulation Experiment")
    print("=" * 60)

    # Get configuration (with override functionality)
    # Use high-precision settings for simulation experiments (specify config_name="simulation")
    # or explicitly override with overrides
    if config_name == "default" and overrides is None:
        # Use high-precision settings for simulation by default
        config = get_config("simulation", overrides)
    else:
        config = get_config(config_name, overrides)
    _print_simulation_config(config)

    # Generate independent seeds for each iteration
    iteration_seeds = _generate_iteration_seeds(
        config.random_seed, config.n_simulations
    )

    # List to store simulation results
    simulation_results = []
    df_list = []  # Save dataframes for each simulation
    neighbors_list_list = []  # Save neighbor lists for each simulation

    print(f"\nRunning simulations...")

    # Display message once if using bootstrap standard errors
    if config.use_bootstrap_se:
        n_jobs = max(1, multiprocessing.cpu_count() // 2)
        print(
            f"Running standard bootstrap method... (n_bootstrap={config.n_bootstrap})"
        )
        print(f"Parallel execution: using {n_jobs} cores (half of max cores)")

    start_time = datetime.now()

    # Execute simulations in normal loop (with progress bar)
    pbar = tqdm(
        range(config.n_simulations),
        desc="Running simulations",
        unit="iter",
        ncols=80,
        colour="green",
    )
    for i in pbar:
        # Execute simulation (data generation + estimator calculation)
        result, df, neighbors_list = run_single_simulation(
            config.n_units,
            config.k_distance,
            config.space_size,
            config,  # Pass config object with overrides applied
            random_seed=iteration_seeds[i],
            progress_bar=pbar,
        )
        simulation_results.append(result)
        df_list.append(df)
        neighbors_list_list.append(neighbors_list)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nSimulation completed: {duration.total_seconds():.2f} seconds")

    # Convert results to dataframe
    results_df = pd.DataFrame(simulation_results)

    print(f"\nNumber of valid simulations: {len(results_df)}")
    print(f"Result columns: {list(results_df.columns)}")

    true_params_avg = {
        "adtt": results_df["true_adtt"].mean(),
        "aitt": results_df["true_aitt"].mean(),
    }
    print(f"\nAverage true parameters (conditional mean):")
    print(
        f"  Avg True ADTT: {true_params_avg['adtt']:.4f}, Avg True AITT: {true_params_avg['aitt']:.4f}"
    )
    return results_df, true_params_avg, df_list, neighbors_list_list


def run_simulation_from_config(
    config_name: str = "default",
    overrides: Optional[Dict[str, Any]] = None,
    skip_basic_plots: bool = False,
    experiment_id: Optional[str] = None,
    analyze_influence: bool = False,
):
    """
    Function to execute simulation based on config file settings

    Executes simulation and saves results based on execution method in README Section 4.3.

    Args:
        config_name: Base configuration name (default: "default")
        overrides: Dictionary of settings to override (supports dot notation)
        skip_basic_plots: Whether to skip basic plots
        experiment_id: Experiment ID
        analyze_influence: Whether to execute influence function distribution analysis (default: False)

    Returns:
        Tuple: (results_df, evaluation_results, sensitivity_table, true_params, output_dir)
    """
    # Get configuration (with override functionality)
    # Use high-precision settings for simulation experiments (specify config_name="simulation")
    # or explicitly override with overrides
    if config_name == "default" and overrides is None:
        # Use high-precision settings for simulation by default
        config = get_config("simulation", overrides)
    else:
        config = get_config(config_name, overrides)

    # Set output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    cleanup_old_results(output_dir)

    # First data generation (before estimator calculation)
    print("\n" + "=" * 60)
    print("Data Generation")
    print("=" * 60)
    print("Generating first dataset...")

    # Generate independent seeds for each iteration (for first data)
    iteration_seeds = _generate_iteration_seeds(
        config.random_seed, config.n_simulations
    )
    first_seed = iteration_seeds[0]

    # Execute only first data generation
    first_simulation_df, _, _ = generate_data(
        n_units=config.n_units,
        K=config.k_distance,
        space_size=config.space_size,
        dgp_config=config,
        random_seed=first_seed,
    )
    print(f"First dataset generation completed: {len(first_simulation_df)} units")

    # Generate Figure 1 and Figure 3 after data generation, before estimator calculation
    if not skip_basic_plots:
        print("\n" + "=" * 60)
        print("Visualization (after data generation)")
        print("=" * 60)

        # Figure 1: Spillover structure
        spillover_plot_file = os.path.join(output_dir, "spillover_structure.png")
        create_spillover_plot(
            save_path=spillover_plot_file, dgp_config=config, df=first_simulation_df
        )
        print(f"Figure 1 saved: {spillover_plot_file}")

        # Figure 3: Scatter plot of unit placement and treatment status
        units_scatter_file = os.path.join(output_dir, "units_scatter_plot.png")
        create_units_scatter_plot(first_simulation_df, save_path=units_scatter_file)
        print(f"Figure 3 saved: {units_scatter_file}")

    # Execute simulation (all simulations)
    results_df, true_params, df_list, neighbors_list_list = run_simulation_experiment(
        config_name, overrides
    )

    # Save results (with experiment ID in filename)
    experiment_id_suffix = _generate_experiment_id(overrides, experiment_id)
    results_file = os.path.join(
        output_dir, f"simulation_results{experiment_id_suffix}.csv"
    )
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved: {results_file}")

    # Execute evaluation (perspectives ① and ② in README Section 3.2.2)
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    evaluation_results = evaluate_estimators(
        results_df, ESTIMATOR_COLUMNS, SE_COLUMNS, config=config
    )

    # Create empty DataFrame since sensitivity analysis is not executed
    sensitivity_table = pd.DataFrame()

    # Save evaluation results (with experiment ID in filename)
    eval_file = os.path.join(
        output_dir, f"evaluation_results{experiment_id_suffix}.csv"
    )
    evaluation_results.to_csv(eval_file, index=False)
    print(f"\nEvaluation results saved: {eval_file}")

    # Execute visualization (after estimator calculation: Figure 2 and Table 2 only)
    if not skip_basic_plots:
        print("\n" + "=" * 60)
        print("Visualization (after estimator calculation)")
        print("=" * 60)

        # Figure 2: Estimator distribution (Figure 2 in README Section 3.2.2)
        distribution_plot_file = os.path.join(
            output_dir, "estimator_distribution_adtt.png"
        )
        create_distribution_plot_adtt(
            results_df, true_params, save_path=distribution_plot_file
        )
        print(f"✓ Figure 2 (ADTT distribution): {distribution_plot_file}")

        # AITT estimator distribution plot
        aitt_distribution_plot_file = os.path.join(
            output_dir, "estimator_distribution_aitt.png"
        )
        create_distribution_plot_aitt(
            results_df, true_params, save_path=aitt_distribution_plot_file
        )
        print(f"✓ Figure (AITT distribution): {aitt_distribution_plot_file}")

        # Table 2: Coverage rate of confidence intervals
        coverage_file = os.path.join(output_dir, "coverage_rate.csv")
        create_coverage_rate_table(
            evaluation_results, config=config, save_path=coverage_file
        )
        print(f"✓ Table 2 (Coverage rate): {coverage_file}")

        # Bootstrap standard error comparison table and distribution plot
        if any(col.endswith("_se_bootstrap") for col in results_df.columns):
            print("\nGenerating bootstrap standard error visualization...")

            # Create comparison table
            se_comparison_file = os.path.join(
                output_dir, f"se_comparison_table{experiment_id_suffix}.csv"
            )
            create_se_comparison_table(results_df, save_path=se_comparison_file)
            print(f"Standard error comparison table saved: {se_comparison_file}")

            # Create distribution plot
            bootstrap_se_dist_file = os.path.join(
                output_dir, f"bootstrap_se_distribution{experiment_id_suffix}.png"
            )
            create_bootstrap_se_distribution_plot(
                results_df, save_path=bootstrap_se_dist_file
            )
            print(
                f"Bootstrap standard error distribution plot saved: {bootstrap_se_dist_file}"
            )
        else:
            print(
                "Warning: Bootstrap standard error columns not found. Skipping visualization."
            )

        # Influence function distribution plot (generated at timing of Coverage_Rate table creation)
        if analyze_influence:
            print("\nCollecting influence function distributions...")
            # Compile results from each simulation into a list
            results_list = [row.to_dict() for _, row in results_df.iterrows()]

            try:
                influence_df = collect_influence_functions_from_simulation(
                    results_list=results_list,
                    df_list=df_list,
                    neighbors_list_list=neighbors_list_list,
                    config=config,
                    dgp_config=config,
                    exposure_type="cs",
                )

                # Visualization
                influence_plot_file = os.path.join(
                    output_dir, "influence_function_comparison.png"
                )
                create_influence_function_comparison_plot(
                    influence_df, save_path=influence_plot_file
                )
                print(
                    f"Influence function distribution plot saved: {influence_plot_file}"
                )

                # Also save CSV
                influence_csv_file = os.path.join(
                    output_dir, "influence_function_data.csv"
                )
                influence_df.to_csv(influence_csv_file, index=False)
                print(f"Influence function data saved: {influence_csv_file}")
            except Exception as e:
                print(
                    f"Warning: Influence function distribution plot generation failed: {e}"
                )
                import traceback

                traceback.print_exc()

    print("\nSimulation experiment completed.")

    return results_df, evaluation_results, sensitivity_table, true_params, output_dir
