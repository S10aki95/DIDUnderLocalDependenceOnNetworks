"""
Common visualization utilities module

Provides common functionality for font settings, plot styles, data extraction, etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, Any, Optional, Tuple, List

# Set matplotlib backend to avoid GUI issues on Windows
matplotlib.use("Agg")

# Japanese font settings
try:
    # Japanese font settings for Windows environment
    font_paths = [
        "C:/Windows/Fonts/msgothic.ttc",  # MS Gothic
        "C:/Windows/Fonts/meiryo.ttc",  # Meiryo
        "C:/Windows/Fonts/yuanti.ttc",  # Yuanti
    ]

    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                plt.rcParams["font.family"] = fm.FontProperties(
                    fname=font_path
                ).get_name()
                break
        except:
            continue

    # Fallback: use default font
    if (
        "font.family" not in plt.rcParams
        or plt.rcParams["font.family"] == "DejaVu Sans"
    ):
        plt.rcParams["font.family"] = "DejaVu Sans"
        print("Warning: Japanese font not found. Using default font.")

except Exception as e:
    print(f"Warning: Error occurred in font settings: {e}")
    plt.rcParams["font.family"] = "DejaVu Sans"


def extract_estimator_data(
    results_data: Dict[str, pd.DataFrame],
    x_values: List[Any],
    metric: str,
    estimator_name: str,
    actual_name: str,
) -> List[float]:
    """Common function to extract estimator data from evaluation results

    Args:
        results_data: Dictionary of result dataframes for each setting (DataFrame loaded from CSV)
        x_values: List of X-axis values
        metric: Evaluation metric ("Bias" or "RMSE")
        estimator_name: Estimator name for display (e.g., "Proposed IPW (ADTT)")
        actual_name: Actual estimator name in CSV file (e.g., "Proposed IPW ADTT (Logistic)")
    """
    values = []
    # Match CSV file column names (lowercase)
    metric_col = metric.lower()  # "Bias" -> "bias", "RMSE" -> "rmse"

    for x_val in x_values:
        if x_val in results_data:
            eval_data = results_data[x_val]
            # Search in estimator column of CSV file (lowercase)
            # Support both uppercase and lowercase
            if "estimator" in eval_data.columns:
                estimator_col = "estimator"
            elif "Estimator" in eval_data.columns:
                estimator_col = "Estimator"
            else:
                values.append(0)
                continue

            matching_rows = eval_data[
                eval_data[estimator_col].str.contains(
                    actual_name, na=False, regex=False
                )
            ]
            if not matching_rows.empty:
                metric_value = matching_rows.iloc[0].get(metric_col, np.nan)
                values.append(metric_value if not pd.isna(metric_value) else 0)
            else:
                values.append(0)
        else:
            values.append(0)
    return values


def setup_plot_style(figsize: Tuple[int, int] = (10, 6)) -> None:
    """Common function to set basic plot style"""
    plt.figure(figsize=figsize)


def save_plot(save_path: Optional[str], plot_type: str = "Figure") -> None:
    """Common function to save plot"""
    if save_path:
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            # Check if file was created
            if not os.path.exists(save_path):
                print(f"Warning: Figure file was not created: {save_path}")
        except Exception as e:
            print(f"Error saving figure to {save_path}: {e}")
            raise
    else:
        print("Warning: save_path is None, figure will not be saved")
        plt.show()
    plt.close()
