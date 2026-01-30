"""
Real data loading and preprocessing functionality

Data loader for analysis using real data from China's SEZ policy
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings("ignore")


class SEZDataLoader:
    """Class for loading and preprocessing real data"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.data = {}

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load real data"""
        print("Loading real data...")

        # Basic datasets
        self.data["village_census"] = pd.read_stata(
            f"{self.data_dir}/village_census.dta"
        )
        self.data["village_above"] = pd.read_stata(f"{self.data_dir}/village_above.dta")
        self.data["county_census"] = pd.read_stata(f"{self.data_dir}/county_census.dta")
        self.data["county_above"] = pd.read_stata(f"{self.data_dir}/county_above.dta")

        # Data for spillover effect analysis
        self.data["village_census_spillover"] = pd.read_stata(
            f"{self.data_dir}/village_census_spillover.dta"
        )
        self.data["village_above_spillover"] = pd.read_stata(
            f"{self.data_dir}/village_above_spillover.dta"
        )

        # Data for heterogeneity analysis
        self.data["village_census_heter_size"] = pd.read_stata(
            f"{self.data_dir}/village_census_heter_size.dta"
        )
        self.data["village_census_heter_kl"] = pd.read_stata(
            f"{self.data_dir}/village_census_heter_kl.dta"
        )
        self.data["village_census_heter_infra"] = pd.read_stata(
            f"{self.data_dir}/village_census_heter_infra.dta"
        )

        print("✓ Real data loading completed")
        return self.data

    def prepare_analysis_data(self, use_spillover_data: bool = True) -> pd.DataFrame:
        """Prepare data for analysis"""
        if use_spillover_data:
            # Use data for spillover effect analysis
            df = self.data["village_census_spillover"].copy()
            print("Using data for spillover effect analysis")
        else:
            # Use basic data
            df = self.data["village_census"].copy()
            print("Using basic data")

        # Display basic data information
        print(f"Data shape: {df.shape}")
        print(f"Years: {sorted(df['year'].unique())}")
        print(f"Number of villages: {df['village'].nunique()}")
        print(f"Number of counties: {df['cty'].nunique()}")

        return df

    def create_panel_data(self, outcome_var: str = "y") -> pd.DataFrame:
        """
        Create panel data (cross-section of 2004 and 2008)

        Args:
            outcome_var: Outcome variable name. One of 'k' (capital), 'l' (employment), 'y' (output)

        Returns:
            Panel dataframe
        """
        if outcome_var not in ["k", "l", "y"]:
            raise ValueError(
                f"outcome_var must be one of ['k', 'l', 'y'], got '{outcome_var}'"
            )

        # Outcome variable names
        outcome_names = {"k": "Capital", "l": "Employment", "y": "Output"}
        outcome_name = outcome_names[outcome_var]

        df_2004 = self.data["village_census"][
            self.data["village_census"]["year"] == 2004
        ].copy()
        df_2008 = self.data["village_census"][
            self.data["village_census"]["year"] == 2008
        ].copy()

        # Merge by village ID to create panel data
        df_panel = pd.merge(
            df_2004, df_2008, on="village", suffixes=("_2004", "_2008"), how="inner"
        )

        # Select and rename necessary variables
        df_analysis = pd.DataFrame(
            {
                "village": df_panel["village"],
                "cty": df_panel["cty_2004"],  # County ID (2004)
                "Y1": df_panel[f"{outcome_var}_2004"],  # Outcome variable in 2004 (log)
                "Y2": df_panel[f"{outcome_var}_2008"],  # Outcome variable in 2008 (log)
                "W": df_panel["sez_2008"],  # SEZ designation flag (2008)
                "airport_p": df_panel["airport_p_2004"],  # Distance from airport (2004)
                "port_p": df_panel["port_p_2004"],  # Distance from port (2004)
                "kl_p": df_panel["kl_p_2004"],  # Capital-labor ratio (2004)
                "num_p": df_panel["num_2004"],  # Number of enterprises (2004)
            }
        )

        # Remove missing values
        df_analysis = df_analysis.dropna()

        return df_analysis

    def create_exposure_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create exposure mapping (G) (optimized version)"""
        df = df.copy()

        # Calculate SEZ village count and total village count per county (use transform to avoid merge)
        df["C_W_sum"] = df.groupby("cty")["W"].transform("sum")
        df["C_Total"] = df.groupby("cty")["W"].transform("count")

        # Calculate Leave-one-out SEZ ratio
        df["LOO_SEZ_Ratio"] = np.where(
            df["C_Total"] > 1,
            (df["C_W_sum"] - df["W"]) / (df["C_Total"] - 1),
            0,  # If there is only one village in the county
        )

        # Calculate threshold (mean of LOO ratios across all villages)
        mean_ratio_threshold = df["LOO_SEZ_Ratio"].mean()

        # Define G (strongly exposed=1, weakly exposed=0)
        df["G"] = (df["LOO_SEZ_Ratio"] > mean_ratio_threshold).astype(int)

        # Remove intermediate columns (memory efficiency)
        df = df.drop(columns=["C_W_sum", "C_Total"])

        return df

    def prepare_covariates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare covariates"""
        df = df.copy()

        # Basic covariates
        Z_list = ["airport_p", "port_p", "kl_p", "num_p"]

        # Calculate Leave-one-out mean values at county level
        # As specified in the paper, calculate LOO mean values excluding the village's own value
        for z in Z_list:
            # Calculate sum and count per county
            county_stats = df.groupby("cty")[z].agg(["sum", "count"])
            county_stats.columns = [f"{z}_sum", f"{z}_count"]
            df = df.merge(county_stats, left_on="cty", right_index=True, how="left")

            # Calculate LOO mean: (Sum - Z_i) / (Count - 1)
            df[f"{z}_county_mean"] = np.where(
                df[f"{z}_count"] > 1,
                (df[f"{z}_sum"] - df[z]) / (df[f"{z}_count"] - 1),
                0.0,  # Set to 0 if there is only one village in the county
            )

            # Remove temporary columns
            df = df.drop(columns=[f"{z}_sum", f"{z}_count"])

        Z_LOO_Mean_list = [f"{z}_county_mean" for z in Z_list]

        # Covariate list for propensity score estimation
        PS_covariates = Z_list + Z_LOO_Mean_list

        return df, PS_covariates

    def get_data_summary(self) -> Dict[str, any]:
        """Get data summary"""
        summary = {}

        for name, df in self.data.items():
            summary[name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "years": sorted(df["year"].unique()) if "year" in df.columns else None,
                "villages": (
                    df["village"].nunique() if "village" in df.columns else None
                ),
                "counties": df["cty"].nunique() if "cty" in df.columns else None,
            }

        return summary


def load_sez_data(data_dir: str = "data") -> SEZDataLoader:
    """Create SEZ data loader instance and load data"""
    loader = SEZDataLoader(data_dir)
    loader.load_data()
    return loader
