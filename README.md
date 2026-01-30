# Difference-in-Differences with Local Dependence on Networks

This repository implements methods for Difference-in-Differences (DID) estimation that account for interference (spillover effects) in treatment assignment. The project compares proposed nonparametric estimators (ADTT/AITT) with Xu (2025)'s doubly robust/IPW estimators and standard DID methods that ignore interference.

For detailed implementation information on simulation design and real data analysis methods, see [src/README.md](src/README.md).

## Project Structure

```
DIDUnderLocalDependenceOnNetworks/
├── src/                    # Source code
│   ├── model/             # Estimator implementations
│   │   ├── proposed/      # Proposed ADTT/AITT estimators
│   │   ├── xu/            # Xu (2025) DR/IPW estimators
│   │   ├── standard/      # Standard DID estimators
│   │   └── bootstrap/     # Bootstrap methods
│   ├── run/               # Execution modules
│   │   ├── simulation.py  # Simulation execution
│   │   └── real_data.py   # Real data analysis execution
│   ├── settings/          # Configuration and settings
│   ├── visualization/     # Visualization and report generation
│   └── utils.py           # Utility functions
├── main.py                # Main execution script
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

For detailed implementation documentation, see [src/README.md](src/README.md).

## Environment Setup

### Required Python Version

- Python 3.12 or higher

### Dependency Installation

```bash
# Clone the project
git clone <repository-url>
cd DIDUnderLocalDependenceOnNetworks

# Install dependencies (using uv)
uv sync
```

**Note**: This project is managed with `uv`. Use `uv run` command when executing.

## Execution Methods

### Command Line Options

| Option                | Default    | Description                                                             |
| --------------------- | ---------- | ----------------------------------------------------------------------- |
| `--mode`              | simulation | Execution mode (simulation or real_data)                                |
| `--data_dir`          | data       | Real data directory (when in real_data mode)                            |
| `--output_dir`        | results    | Output directory                                                        |
| `--config`            | default    | Configuration name (simulation mode)                                    |
| `--use_bootstrap`     | False      | Enable bootstrap standard error calculation (common to both modes)      |
| `--analyze_influence` | False      | Enable influence function distribution analysis (simulation mode)       |
| `--opt_band`          | False      | Enable bandwidth optimization for HAC standard errors (simulation mode) |

### Simulation Experiments

```bash
# Run simulation experiment (basic experiment + robustness experiment)
uv run python main.py --mode simulation

# Run simulation with bootstrap standard errors enabled
uv run python main.py --mode simulation --use_bootstrap

# Run simulation with bandwidth optimization
uv run python main.py --mode simulation --opt_band

# Single experiment with specific configuration (optional)
uv run python main.py --mode simulation --config default
```

For detailed simulation design and estimator implementation, see [src/README.md](src/README.md).

### Real Data Experiments

```bash
# Run real data experiment (default settings)
uv run python main.py --mode real_data

# Specify data directory and output directory
uv run python main.py --mode real_data --data_dir data --output_dir results
```

For detailed real data analysis methods and data description, see [src/README.md](src/README.md).

### Bandwidth Optimization (Optional Feature)

In simulation experiments, bandwidth optimization can be performed for HAC standard error calculation used in proposed methods.

```bash
# Run simulation with bandwidth optimization
uv run python main.py --mode simulation --opt_band
```

**Note:** Bandwidth optimization is only available in simulation mode. HAC standard errors are not used in real data mode because spatial coordinate information is not available.

## Output Files

Results are saved in the `results/` directory (or the directory specified by `--output_dir`).

### Simulation Results

- `simulation_results.csv`: Simulation results data (estimates and standard errors for each trial)
- `evaluation_results.csv`: Estimator evaluation results (Bias, RMSE, Coverage Rate)
- `robustness_results.csv`: Integrated results of robustness experiments
- `robustness_summary.md`: Summary report of all experiment results
- `spillover_structure.png`: Spillover effect count distribution
- `estimator_distribution_adtt.png`: Distribution of each estimator
- `units_scatter_plot.png`: Unit placement and treatment status
- `robustness_*.png`: Robustness analysis charts
- `sensitivity_*.png`: Sensitivity analysis charts

### Real Data Results

- `sez_analysis_results.csv`: Analysis results (estimates and standard errors)
- `sez_analysis_report.md`: Detailed results report
- `missing_data_analysis.csv`: Missing data situation

For detailed descriptions of each output file, see [src/README.md](src/README.md).

## References

- Xu, R. (2025). Difference-in-Differences with Interference. *arXiv preprint arXiv:2306.12003*.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *The Review of Economic Studies*, 72(1), 1-19.
- Kojevnikov, D., Marmer, V., & Song, K. (2021). Limit theorems for network dependent random variables. *Journal of Econometrics*, 222(2), 882-908.
