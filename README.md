# HST Project 1 Python Code Documentation

This document explains the Python code in this repository for readers who are new to portfolio optimisation, CVaR, Gurobi, or metaheuristic algorithms. It is written as a practical guide to what each file does, how data moves through the project, and which files should be run for which purpose.

## Big Picture

The project solves a constrained portfolio optimisation problem. The goal is to choose a small portfolio from 100 assets and assign weights so that the portfolio has:

- high annualised expected return,
- low downside tail risk, measured by historical CVaR,
- at most 15 selected assets,
- valid position sizes,
- enough bond exposure,
- limited crypto exposure,
- limited sector concentration.

The code has two main optimisation paths:

1. `gurobi_optimizer.py` solves an exact mixed-integer optimisation model for different CVaR limits. This gives a reference Pareto front.
2. `nsga2_optimizer.py` defines metaheuristic problem classes used by NSGA-II, MOEA/D, and AGE-MOEA through `pymoo`.

The main entry point is `src/main_pipeline.py`. It runs the full experiment and saves all important numbers to `pipeline_results.json`. The plotting entry point is `src/generate_report_figures.py`, which reads the JSON file and creates figures in `figures/`.

## Data Flow

The usual flow is:

1. `config.py` defines tickers, dates, constraints, and sectors.
2. `fetch_universe.py` loads cached adjusted close prices from `src/raw/prices.csv`, or downloads them with `yfinance` if the cache is missing.
3. `compute_return.py` converts prices into daily log returns, annualised mean returns, and covariance.
4. `compute_cvar.py` evaluates portfolio return and CVaR.
5. `gurobi_optimizer.py` and `nsga2_optimizer.py` optimise portfolios.
6. `run_gurobi_scaling_frontier.py` runs the full-frontier Gurobi scaling experiment.
7. `stress_window.py` evaluates stress windows.
8. `main_pipeline.py` saves results to `pipeline_results.json`.
9. `generate_report_figures.py` turns the JSON results into figures.

## File-by-File Explanation

### `src/config.py`

This is the central settings file. Almost every other Python file imports values from it.

Important variables:

- `M = 100`: number of assets in the full universe.
- `K = 15`: maximum number of holdings allowed in the portfolio.
- `START_DATE` and `END_DATE`: requested data range for price collection.
- `COVID_START`, `COVID_END`, `TARIFF_START`, `TARIFF_END`: stress-test windows.
- `W_MIN = 0.02`: if an asset is selected, it must receive at least 2 percent weight.
- `W_MAX = 0.30`: no selected asset can receive more than 30 percent weight.
- `SECTOR_CAP = 0.40`: maximum sector exposure.
- `CRYPTO_CAP = 0.20`: maximum crypto exposure.
- `BOND_FLOOR = 0.10`: minimum bond allocation.
- `CVAR_ALPHA = 0.95`: CVaR confidence level.
- `SP500_EQUITIES`, `BOND_ETFS`, `BROAD_ETFS`, `CRYPTOS`: asset groups.
- `ALL_TICKERS`: the ordered list of all 100 assets.
- `SECTOR_MAP`: maps each ticker to a sector or asset class.

Why this file matters:

Keeping these settings in one place avoids hidden inconsistencies. For example, if the bond floor changes from 10 percent to 15 percent, it should be changed here rather than separately inside every algorithm.

### `src/fetch_universe.py`

This file loads the price data.

Main function:

```python
fetch_prices(tickers=ALL_TICKERS, start=START_DATE, end=END_DATE, cache=True, cache_name="prices.csv")
```

What it does:

1. Looks for `src/raw/prices.csv`.
2. If the file exists and `cache=True`, it loads prices from that CSV.
3. If the cache is missing, it downloads adjusted prices from Yahoo Finance using `yfinance`.
4. It drops assets with no data.
5. It forward-fills small missing gaps.
6. It drops remaining missing rows so the return matrix is rectangular.
7. It saves the downloaded data to the cache.

Important note:

The actual final date range can be shorter than `START_DATE` to `END_DATE`. This happens because all 100 assets must have data on the same days. Since some crypto assets start later, the final aligned dataset begins later than the requested start date.

### `src/compute_return.py`

This file transforms prices into return inputs.

Functions:

- `log_returns(prices)`: computes daily log returns using `log(P_t / P_{t-1})`.
- `annualised_mean(ret)`: multiplies average daily returns by 252 trading days.
- `covariance_matrix(ret)`: computes the annualised covariance matrix.
- `compute_all(prices)`: returns all three objects at once: `ret`, `mu`, and `Sigma`.

What the outputs mean:

- `ret`: historical daily return scenarios used for CVaR.
- `mu`: expected annual return for each asset.
- `Sigma`: annualised covariance matrix. The current optimisation focuses on CVaR, but covariance is still useful for analysis or extensions.

### `src/compute_cvar.py`

This file evaluates the objective values for a portfolio.

Functions:

- `portfolio_returns(weights, scenarios)`: calculates daily portfolio returns by multiplying the scenario matrix by the weight vector.
- `cvar(weights, scenarios, alpha=CVAR_ALPHA)`: calculates historical CVaR as a positive loss.
- `expected_return(weights, mu)`: calculates annualised portfolio expected return.

How CVaR is computed:

1. Compute daily portfolio returns.
2. Find the worst 5 percent of returns when `alpha = 0.95`.
3. Average those bad returns.
4. Negate the result so that CVaR is reported as a positive loss.

Interpretation:

Lower CVaR is better because it means smaller average loss in the worst historical days.

### `src/constraints.py`

This file checks and repairs portfolios.

Main functions:

- `get_sector_indices(tickers)`: groups ticker indices by sector.
- `is_feasible(z, w, tickers)`: checks whether a portfolio satisfies all constraints.
- `repair(z, w, tickers)`: tries to convert an infeasible portfolio into a more feasible one.

Key variables:

- `z`: binary selection vector. `z[i] = 1` means asset `i` is selected.
- `w`: weight vector. `w[i]` is the portfolio weight assigned to asset `i`.

What `is_feasible()` checks:

- weights sum to 1,
- selected assets do not exceed `K`,
- selected assets satisfy weight bounds,
- unselected assets have zero weight,
- ordinary equity sectors do not exceed the sector cap,
- crypto does not exceed the crypto cap,
- bonds meet the bond floor.

What `repair()` does:

The metaheuristics often produce invalid candidate portfolios. `repair()` tries to make them usable by:

1. dropping assets if more than `K` are selected,
2. setting unselected weights to zero,
3. normalising selected weights,
4. clipping selected weights between `W_MIN` and `W_MAX`,
5. ensuring at least one bond is selected when possible,
6. increasing bond weight toward the bond floor,
7. reducing crypto weight if above the cap,
8. normalising again.

Important caveat:

`repair()` is a heuristic projection, not a guaranteed mathematical optimiser. The saved `pipeline_results.json` currently records one random repair smoke test with a small bond-floor violation. The selected Gurobi portfolio itself passes the saved constraint verification, but Repair should be checked carefully before treating it as deployment-ready.

### `src/gurobi_optimizer.py`

This file implements the exact optimisation model using Gurobi.

Main functions:

- `_build_cvar_model(...)`: builds one Gurobi optimisation model for a fixed CVaR limit.
- `optimize_gurobi(mu, scenarios, tickers, epsilon_values=None, verbose=False)`: solves many Gurobi models across different CVaR limits to trace the Pareto frontier.
- `scaling_experiment_proper(mu, scenarios, tickers, n_epsilons=20, time_limit=300)`: older standalone scaling helper retained for reference. The main pipeline now uses `run_gurobi_scaling_frontier.py` for the report-facing scaling experiment.

Decision variables:

- `w_i`: portfolio weight for asset `i`.
- `z_i`: binary variable showing whether asset `i` is selected.
- `VaR`: auxiliary Value-at-Risk variable.
- `U_t`: auxiliary shortfall variable for day `t`.

Main optimisation idea:

The true project has two objectives:

- maximise return,
- minimise CVaR.

Gurobi solves this by fixing a CVaR upper bound `epsilon` and maximising return. Repeating this for many epsilon values gives multiple points on the Pareto front.

CVaR linearisation:

CVaR is represented using the Rockafellar-Uryasev linear formulation. This makes the historical CVaR objective compatible with mixed-integer linear optimisation.

Important consistency note:

The Gurobi model applies sector caps only to ordinary equity sectors and separately handles bond and crypto limits. The feasibility checker, penalty strategy, and decoder use the same interpretation. Broad ETFs are not given a separate group cap unless one is added explicitly.

### `src/run_gurobi_scaling_frontier.py`

This file runs the report-facing Gurobi scaling experiment.

What it does:

1. Loads `src/raw/prices.csv`.
2. Builds sub-universes for `M = 20, 40, 60, 80, 100, 150, 200`, skipping sizes larger than the available price panel.
3. Solves a full 20-point epsilon frontier for each available universe size using `EPS_GRID = np.linspace(0.010, 0.045, 20)`.
4. Records runtime, feasibility, optimality, MIP gap, branch-and-bound nodes, best return, and lowest realised CVaR.
5. Saves raw and summary scaling files under `scaling_results/`.

Generated files:

- `scaling_results/gurobi_scaling_full_frontier_raw.csv`
- `scaling_results/gurobi_scaling_full_frontier_summary.csv`
- `scaling_results/gurobi_scaling_full_frontier_summary.json`

Important settings:

- `TIME_LIMIT_PER_EPS = 300`: each epsilon solve may run for up to 300 seconds.
- `MIP_GAP_TARGET = 1e-4`: Gurobi target optimality gap.
- `EPS_GRID`: fixed 20-point CVaR grid from `0.010` to `0.045`.

Important caveat:

This experiment is empirical solver evidence, not a mathematical proof of NP-hardness. It is useful for showing how exact mixed-integer optimisation behaves as the universe grows, but runtime depends on the data, epsilon grid, Gurobi settings, and hardware.

### `src/nsga2_optimizer.py`

This file defines the portfolio problem classes used by `pymoo` algorithms.

Classes:

- `BasePortfolioProblem`: shared setup and local weight solver.
- `PortfolioProblem`: Repair strategy.
- `PortfolioProblemPenalty`: Penalty strategy.
- `PortfolioProblemDecoder`: Decoder strategy.

Why this file exists:

Algorithms like NSGA-II need a standard problem object. Each candidate solution is evaluated by converting it into selected assets and weights, then computing:

- objective 1: negative expected return,
- objective 2: CVaR.

The return is negated because `pymoo` minimises objectives by default. Minimising negative return is equivalent to maximising return.

How weights are chosen:

For a fixed selected asset set, `_solve_weights()` uses SLSQP, a local continuous optimiser, to assign weights. This combines a discrete search over asset selection with a continuous optimisation over weights.

Constraint strategies:

1. Repair:
   - evolves binary `z` vectors,
   - solves weights for selected assets,
   - calls `constraints.repair()`.

2. Penalty:
   - evolves binary `z` vectors,
   - allows infeasible candidates,
   - adds penalty values to the objectives when constraints are violated.

3. Decoder:
   - evolves continuous priority scores in `[0, 1]`,
   - selects the top `K` assets by score,
   - forces at least one bond when possible,
   - applies post-processing for bond, crypto, and sector limits.

### `src/stress_window.py`

This file extracts stress-test windows and computes realised performance metrics.

Functions:

- `tariff_window(ret_full)`: extracts the April 2025 tariff shock window.
- `covid_window(ret)`: extracts the COVID-19 window.
- `stress_metrics(weights, ret_window, mu)`: calculates realised CVaR, maximum drawdown, and Sharpe ratio.

Metrics:

- realised CVaR: tail loss during the stress window,
- maximum drawdown: largest peak-to-trough loss during the window,
- Sharpe ratio: annualised return divided by annualised volatility over the stress window.

Note:

The `mu` argument is present for API compatibility, but the realised stress metrics are computed from actual stress-window returns.

### `src/main_pipeline.py`

This is the main experiment script. It should be run when you want to regenerate the complete study.

Main sections:

1. Load prices and compute returns.
2. Evaluate the equal-weight baseline.
3. Stress-test the equal-weight baseline.
4. Run Gurobi exact optimisation.
5. Run the Gurobi full-frontier scaling experiment.
6. Compare constraint-handling strategies.
7. Compare search operators.
8. Compare NSGA-II, MOEA/D, AGE-MOEA, and Gurobi.
9. Run statistical testing across 15 independent seeds.
10. Generate 100-generation convergence curves.
11. Analyse selected portfolio weights and sector exposure.
12. Stress-test the selected Gurobi portfolio.
13. Save all results to `pipeline_results.json`.

Important constants:

- `N_GEN_MAIN = 50`: generations for main metaheuristic comparisons.
- `N_GEN_CONVERGE = 100`: generations for convergence plots.
- `N_RUNS_STATS = 15`: independent runs for statistical testing.
- `POP_SIZE = 500`: metaheuristic population size.
- `SEED = 42`: default random seed.

Important output keys:

- `data`
- `baseline`
- `repair_sanity_check`
- `stress_equal_weight`
- `gurobi_pareto_front`
- `best_portfolio`
- `scaling`
- `scaling_details`
- `scaling_config`
- `constraint_handling`
- `constraint_handling_curves`
- `search_operators`
- `search_operator_curves`
- `algorithm_comparison`
- `hv_stats`
- `mann_whitney`
- `convergence_100gen`
- `portfolio_allocation`
- `sector_breakdown`
- `constraint_verification`
- `stress_optimised`

Runtime warning:

This script can take a long time because it runs Gurobi, the full-frontier scaling experiment, many metaheuristic optimisations, repeated statistical runs, and convergence tracking. The scaling section alone may solve 20 Gurobi models per available universe size, with a 300-second time limit per epsilon point.

### `src/generate_report_figures.py`

This script creates report figures from `pipeline_results.json`.

What it does:

1. Loads `pipeline_results.json`.
2. Extracts saved Gurobi, metaheuristic, statistical, stress-test, and portfolio data.
3. Creates figures under `figures/`.

Main plot functions:

- `plot_scaling()`: Gurobi full-frontier total runtime as universe size increases.
- `plot_pareto_front()`: exact Gurobi CVaR-return Pareto front.
- `plot_constraint_handling_convergence()`: HV curves for Repair, Penalty, and Decoder.
- `plot_operator_convergence()`: HV curves for crossover operators.
- `plot_algorithm_convergence()`: HV curves for NSGA-II, MOEA/D, and AGE-MOEA.
- `plot_config_summary()`: bar charts comparing configuration choices.
- `plot_algorithm_summary()`: final HV and runtime comparison.
- `plot_statistical_boxplot()`: HV distributions and Mann-Whitney U results.
- `plot_stress_test()`: equal-weight versus optimised stress metrics.
- `plot_portfolio_weights()`: selected portfolio allocation chart.

Important note:

This script does not rerun optimisation. It only visualises results already saved by `main_pipeline.py`.

### `src/stress_testing.py`

This is a standalone stress-test figure helper.

What it does:

1. Loads prices and returns.
2. Loads the saved Gurobi portfolio from `pipeline_results.json`.
3. Builds equal-weight and Gurobi portfolio weights.
4. Optionally reruns faster metaheuristic approximations.
5. Creates additional stress-test bar charts and cumulative return curves.

Important caveat:

The metaheuristic weight reconstruction in this helper is approximate. It selects top-K assets from final populations and assigns equal weights for stress-testing convenience. That is not exactly the same as the SLSQP-optimised weights used inside `nsga2_optimizer.py`. Therefore, treat the metaheuristic stress-test outputs from this standalone helper as supplementary unless exact metaheuristic weight vectors are saved from the main pipeline.

### `src/rerun_constraint_handling_convergence.py`

This helper reruns only the constraint-handling convergence curves.

Use it when:

- `main_pipeline.py` has already produced `pipeline_results.json`,
- but you need to regenerate the Repair/Penalty/Decoder HV curves,
- and you do not want to rerun the full multi-hour pipeline.

What it does:

1. Loads the existing JSON results.
2. Recomputes returns from cached prices.
3. Rebuilds the three constraint-handling problem classes.
4. Runs NSGA-II for each method.
5. Records HV and elapsed time at each generation.
6. Backs up the old JSON.
7. Writes updated `constraint_handling_curves` into `pipeline_results.json`.

### Notebooks

The repository also includes notebooks:

- `sector.ipynb`
- `src/data.ipynb`

These appear to be exploratory notebooks used for data screening and sector/universe construction. They are not part of the automated pipeline, but they help explain where the selected equity universe came from.

## Consistency Notes From Code Review

The current codebase checks include:

- all Python files compile successfully,
- `ALL_TICKERS` contains 100 unique tickers,
- every ticker in `ALL_TICKERS` now has a sector mapping,
- there are no extra sector mappings for tickers outside the universe,
- `generate_report_figures.py` reads the saved `pipeline_results.json` and writes report figures to `figures/`.

Modelling notes:

- `repair()` is heuristic and can occasionally leave tiny feasibility issues, especially around the bond floor after renormalisation.
- Broad ETFs are not group-capped by the current mathematical model. They are only limited by individual position bounds unless an explicit ETF cap is added.
- The standalone `stress_testing.py` script approximates metaheuristic stress weights unless exact metaheuristic weights are saved.
- The full-frontier scaling experiment writes extra CSV/JSON files under `scaling_results/`; these are separate from `pipeline_results.json`.

## Recommended Run Order

For a normal final report refresh:

```bash
python src/main_pipeline.py
python src/generate_report_figures.py
```

For only regenerating the constraint-handling convergence curves:

```bash
python src/rerun_constraint_handling_convergence.py
python src/generate_report_figures.py
```

For additional stress-test figures:

```bash
python src/stress_testing.py
```

For only the Gurobi full-frontier scaling experiment:

```bash
python src/run_gurobi_scaling_frontier.py
```

