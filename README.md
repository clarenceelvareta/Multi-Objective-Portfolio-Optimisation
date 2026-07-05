# HST Project 1: Multi-Objective Portfolio Optimisation

This repository studies a constrained multi-asset portfolio optimisation
problem with two objectives:

- maximise annualised expected return,
- minimise historical CVaR tail risk.

The project compares an exact Gurobi mixed-integer formulation against
metaheuristic algorithms implemented with pymoo: NSGA-II, MOEA/D, and
AGE-MOEA. The portfolio includes S&P 500 equities, bond ETFs, broad ETFs,
and cryptocurrencies, with constraints on cardinality, asset weights,
sector exposure, crypto exposure, and bond allocation.

## Important Runtime Note

`src/main_pipeline.py` is long-running. It performs Gurobi optimisation,
scaling experiments, repeated stochastic algorithm runs, convergence
curves, stress tests, and JSON export. Do not start another full pipeline
run while one is already running unless you intentionally want to restart
or duplicate the experiment.

## Project Structure

- `src/config.py`  
  Central configuration: ticker universe, date range, stress windows,
  constraints, and sector mappings.

- `src/fetch_universe.py`  
  Loads adjusted close prices from `src/raw/prices.csv` when available,
  or downloads them from yfinance.

- `src/compute_return.py`  
  Converts prices into daily log returns, annualised mean returns, and
  annualised covariance.

- `src/compute_cvar.py`  
  Computes portfolio scenario returns, expected return, and historical
  CVaR using a positive-loss convention.

- `src/constraints.py`  
  Checks portfolio feasibility and repairs candidate solutions generated
  by metaheuristics.

- `src/gurobi_optimizer.py`  
  Builds and solves the exact epsilon-constrained mixed-integer CVaR
  model in Gurobi.

- `src/nsga2_optimizer.py`  
  Defines pymoo problem classes for Repair, Penalty, and Decoder
  constraint-handling strategies.

- `src/main_pipeline.py`  
  Runs the full experiment end to end and writes `pipeline_results.json`.

- `src/rerun_test.py`  
  Targeted long-running rerun script for statistical and convergence
  experiments after `pipeline_results.json` already exists.

- `src/generate_report_figures.py`  
  Reads `results_updated.json` and writes report-ready figures to
  `figures/`.

- `src/tets.py`  
  Standalone stress-test figure helper for additional robustness plots.

- `sector.ipynb` and `src/data.ipynb`  
  Notebook-based data exploration and equity screening.

## Dependencies

The project expects a Python environment with at least:

- numpy
- pandas
- scipy
- matplotlib
- yfinance
- pymoo
- gurobipy

Gurobi also requires a working local Gurobi installation and license.

## Typical Workflow

1. Check or update project assumptions in `src/config.py`.
2. Ensure price data is available in `src/raw/prices.csv`, or allow
   `fetch_universe.py` / `main_pipeline.py` to download it.
3. Run the full study:

   ```bash
   python src/main_pipeline.py
   ```

4. If needed, run the targeted rerun script for additional statistical
   or convergence results:

   ```bash
   python src/rerun_test.py
   ```

5. Generate report figures after the relevant JSON file exists:

   ```bash
   python src/generate_report_figures.py
   ```

## Main Outputs

- `pipeline_results.json`  
  Full output from `main_pipeline.py`.

- `results_checkpoint_after_7_5.json`  
  Checkpoint after the statistical rerun section.

- `results_checkpoint_convergence.json`  
  Checkpoint during convergence reruns.

- `results_updated.json`  
  Updated results used by `generate_report_figures.py`.

- `figures/`  
  PDF and PNG plots for the report.


## Notes

- Hypervolume, GD, and IGD must be evaluated using pymoo's minimisation
  convention. The pipeline stores human-readable fronts as
  `(return, CVaR)` but converts them internally to `(-return, CVaR)` for
  pymoo indicators.
- CVaR is reported as a positive loss, so lower CVaR is better.
- The full pipeline is stochastic in the metaheuristic sections. Seeds
  are fixed where possible, but repeated-run statistics are still the
  right way to compare algorithms.
