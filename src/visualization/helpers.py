"""
Helper functions module

Provides helper functions for estimator name mapping, experiment classification, cleanup, etc.
"""

import os
import glob
import pandas as pd
from typing import Dict, Any, List, Tuple


def classify_experiments_by_type(
    all_results: Dict[str, Any],
) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Common helper function to classify experiment results by type

    Args:
        all_results: Results of robustness experiments

    Returns:
        Dict[str, List[Tuple[str, Any]]]: Experiment results classified by type
    """
    experiment_groups = {
        "sample_size": [],
        "density": [],
        "correlation": [],
        "features": [],
    }

    # Classify each experiment result by type
    for experiment_id, result in all_results.items():
        if "error" in result:
            continue  # Skip experiments with errors

        if "evaluation_results" not in result:
            continue  # Skip experiments without evaluation results

        # Determine type from experiment ID
        if "sample_size" in experiment_id:
            experiment_groups["sample_size"].append((experiment_id, result))
        elif "network_density" in experiment_id or "k_" in experiment_id:
            experiment_groups["density"].append((experiment_id, result))
        elif "spatial_correlation" in experiment_id or "correlation" in experiment_id:
            experiment_groups["correlation"].append((experiment_id, result))
        elif "neighbor_features" in experiment_id or "features" in experiment_id:
            experiment_groups["features"].append((experiment_id, result))

    return experiment_groups


def map_estimator_names(evaluation_results: pd.DataFrame) -> pd.DataFrame:
    """
    Function to map estimator display names according to README requirements

    Args:
        evaluation_results: Dataframe of evaluation results

    Returns:
        pd.DataFrame: Dataframe with mapped estimator names
    """
    # Mapping from internal names to display names
    name_mapping = {
        "Proposed IPW ADTT (Logistic)": "Proposed IPW (ADTT)",
        "Proposed IPW AITT (Logistic)": "Proposed IPW (AITT)",
        "Proposed DR ADTT (Logistic)": "Proposed DR (ADTT)",
        "Proposed DR AITT (Logistic)": "Proposed DR (AITT)",
        "Xu DR (CS) - ODE": "Xu DR (CS) - ODE",
        "Xu DR (MO) - ODE": "Xu DR (MO) - ODE",
        "Xu DR (FM) - ODE": "Xu DR (FM) - ODE",
        "Xu IPW (CS) - ODE": "Xu IPW (CS) - ODE",
        "Xu IPW (MO) - ODE": "Xu IPW (MO) - ODE",
        "Xu IPW (FM) - ODE": "Xu IPW (FM) - ODE",
        "Canonical IPW": "Canonical IPW",
        "Modified TWFE": "Modified TWFE",
        "DR-DID": "DR-DID",
    }

    # Copy dataframe and modify
    result_df = evaluation_results.copy()

    # Map values in Estimator column
    if "Estimator" in result_df.columns:
        result_df["Estimator"] = (
            result_df["Estimator"].map(name_mapping).fillna(result_df["Estimator"])
        )

    return result_df


def _generate_estimator_descriptions() -> List[str]:
    """
    Common function to generate estimator description blocks

    Returns:
        List[str]: Markdown content containing estimator descriptions
    """
    return [
        "**Estimator Descriptions:**",
        "- **Proposed IPW ADTT/AITT**: Non-parametric IPW estimator proposed in this study",
        "- **Proposed DR ADTT/AITT**: Non-parametric DR estimator proposed in this study",
        "- **Canonical IPW**: Standard IPW-DID estimator that ignores interference",
        "- **Canonical TWFE**: Standard two-period fixed effects model that ignores interference",
        "- **DR-DID**: Doubly robust difference-in-differences estimator",
        "- **Modified TWFE**: Extended TWFE model that considers interference",
        "- **Xu (CS/MO/FM) - ODE**: Xu (2025) IPW estimator (with different exposure mappings)",
        "",
    ]


def cleanup_old_results(output_dir: str = "results") -> None:
    """
    Function to clean up old timestamped files

    Args:
        output_dir: Path to results directory
    """
    if not os.path.exists(output_dir):
        return

    # Pattern for timestamped files
    patterns = [
        "simulation_results_*.csv",
        "evaluation_results_*.csv",
        "spillover_structure_*.png",
        "estimator_distributions_*.png",
    ]

    removed_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(output_dir, pattern))
        for file_path in files:
            try:
                os.remove(file_path)
                removed_files.append(file_path)
            except OSError as e:
                print(f"Failed to delete file {file_path}: {e}")

    if removed_files:
        print(f"Deleted old files: {len(removed_files)} files")
        for file_path in removed_files:
            print(f"  - {os.path.basename(file_path)}")
    else:
        print("No old files to delete.")
