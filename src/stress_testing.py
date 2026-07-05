"""
Additional financial robustness and stress-test figure script.

This standalone helper loads `results_updated.json`, reconstructs the
Gurobi portfolio, optionally reruns faster versions of the metaheuristics
to obtain representative portfolio weights, and writes stress-test
figures under `figures/`.

Outputs:
    figures/stress_bar_chart.pdf/png
    figures/stress_cumulative.pdf/png

Important:
    This file performs data loading and may rerun optimisation when
    executed. Run it only after the main pipeline has produced the
    required JSON results.
"""

"""
section_5_4.py
==============
Financial Robustness & Stress Testing (Section 5.4)

Run this AFTER main_pipeline.py has produced pipeline_results.json
AND after your NSGA-II / MOEA/D / AGE-MOEA best portfolios are known.

What this produces:
  figures/stress_bar_chart.png     — CVaR / Max Drawdown / Sharpe bar comparison
  figures/stress_cumulative.png    — Cumulative return curves during each shock

Usage:
  python section_5_4.py

All imports are from files you already have. No new dependencies.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fetch_universe import fetch_prices
from compute_return  import compute_all
from stress_window   import covid_window, tariff_window, stress_metrics
from config          import BOND_ETFS, CRYPTOS, BROAD_ETFS, SECTOR_MAP

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load price data (uses cache if available)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading price data...")
prices  = fetch_prices()
prices  = prices.dropna(axis=1, how="all")
tickers = list(prices.columns)
M       = len(tickers)
ret, mu, Sigma = compute_all(prices)
scenarios      = ret.values

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load Gurobi best portfolio from results_updated.json
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_PATH = "results_updated.json"

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(
        f"{RESULTS_PATH} not found. Run main_pipeline.py first."
    )

with open(RESULTS_PATH) as f:
    pipeline = json.load(f)

# Reconstruct Gurobi best weight vector from saved allocation
gurobi_weights_dict = pipeline["best_portfolio"]["weights"]
w_gurobi = np.zeros(M)
for i, t in enumerate(tickers):
    if t in gurobi_weights_dict:
        w_gurobi[i] = gurobi_weights_dict[t]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Define portfolios to stress test
#
# The Gurobi portfolio comes from pipeline_results.json.
# For NSGA-II / MOEA/D / AGE-MOEA best portfolios, the pipeline picks the
# solution with the best realised Sharpe from each algorithm's Pareto front.
# Since those weight vectors aren't saved to JSON by default (only the
# Gurobi weights are), we rerun a short optimisation here to get them.
#
# If you have saved those weight vectors separately, replace the
# "NSGA-II best", "MOEA/D best", "AGE-MOEA best" entries below with
# your actual numpy arrays and skip the re-run block.
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding portfolio weight vectors...")

# Equal-weight baseline (always available)
w_eq = np.ones(M) / M

def best_sharpe_from_front(F_matrix, w_matrix, ret_series, rf=0.02):
    """
    Given a Pareto front F (n_pts x 2) and corresponding weight matrix
    (n_pts x M), return the weight vector with the highest realised Sharpe.
    """
    best_idx, best_s = 0, -np.inf
    for i, w in enumerate(w_matrix):
        port = ret_series @ w
        ann_r = port.mean() * 252
        ann_v = port.std()  * (252 ** 0.5)
        s = (ann_r - rf) / ann_v if ann_v > 0 else -np.inf
        if s > best_s:
            best_s, best_idx = s, i
    return w_matrix[best_idx]


# ── Re-run a quick NSGA-II / MOEA/D / AGE-MOEA (50 gen, pop 200) to get
#    representative Pareto fronts + weight vectors for stress testing.
#    This is much faster than the full 500-pop run.
#    Comment this block out and supply your own w_nsga / w_moead / w_age
#    if you already have the weight vectors.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from nsga2_optimizer import PortfolioProblemDecoder
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.age   import AGEMOEA
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm   import PM
    from pymoo.operators.sampling.rnd  import FloatRandomSampling, IntegerRandomSampling
    from pymoo.optimize                import minimize as pymoo_minimize
    from pymoo.util.ref_dirs           import get_reference_directions

    FAST_POP = 200
    FAST_GEN = 50
    SEED     = 42

    problem = PortfolioProblemDecoder(mu, scenarios, tickers)

    print(f"  Running NSGA-II (pop={FAST_POP}, gen={FAST_GEN})...")
    res_nsga = pymoo_minimize(
        problem,
        NSGA2(pop_size=FAST_POP, sampling=FloatRandomSampling(),
              crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
              eliminate_duplicates=True),
        ("n_gen", FAST_GEN), seed=SEED, verbose=False
    )
    # collect weight vectors from the final population
    pop_X_nsga = res_nsga.pop.get("X")
    w_list_nsga = []
    for x in pop_X_nsga:
        sel = np.argsort(x)[-15:]  # top-K by decoder score
        w_sub = np.zeros(M)
        w_sub[sel] = 1.0 / len(sel)
        w_list_nsga.append(w_sub)
    w_nsga = best_sharpe_from_front(res_nsga.F, np.array(w_list_nsga), ret.values)

    print(f"  Running MOEA/D (pop={FAST_POP}, gen={FAST_GEN})...")
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    res_moead = pymoo_minimize(
        problem,
        MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.9,
              sampling=FloatRandomSampling(),
              crossover=SBX(prob=0.9, eta=15),
              mutation=PM(eta=20)),
        ("n_gen", FAST_GEN), seed=SEED, verbose=False
    )
    pop_X_moead = res_moead.pop.get("X")
    w_list_moead = []
    for x in pop_X_moead:
        sel = np.argsort(x)[-15:]
        w_sub = np.zeros(M)
        w_sub[sel] = 1.0 / len(sel)
        w_list_moead.append(w_sub)
    w_moead = best_sharpe_from_front(res_moead.F, np.array(w_list_moead), ret.values)

    print(f"  Running AGE-MOEA (pop={FAST_POP}, gen={FAST_GEN})...")
    res_age = pymoo_minimize(
        problem,
        AGEMOEA(pop_size=FAST_POP, sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                eliminate_duplicates=True),
        ("n_gen", FAST_GEN), seed=SEED, verbose=False
    )
    pop_X_age = res_age.pop.get("X")
    w_list_age = []
    for x in pop_X_age:
        sel = np.argsort(x)[-15:]
        w_sub = np.zeros(M)
        w_sub[sel] = 1.0 / len(sel)
        w_list_age.append(w_sub)
    w_age = best_sharpe_from_front(res_age.F, np.array(w_list_age), ret.values)

    algo_weights_available = True

except ImportError as e:
    print(f"  pymoo not available ({e}), skipping algorithm re-run.")
    print("  Only Equal-Weight and Gurobi will be stress-tested.")
    w_nsga = w_moead = w_age = None
    algo_weights_available = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Compute stress metrics for each portfolio
# ─────────────────────────────────────────────────────────────────────────────
ret_covid  = covid_window(ret)
ret_tariff = tariff_window(ret)

portfolios = {"Equal Weight": w_eq, "Gurobi": w_gurobi}
if algo_weights_available:
    portfolios["NSGA-II"]  = w_nsga
    portfolios["MOEA/D"]   = w_moead
    portfolios["AGE-MOEA"] = w_age

print("\nComputing stress metrics...")
covid_metrics  = {}
tariff_metrics = {}
for name, w in portfolios.items():
    covid_metrics[name]  = stress_metrics(w, ret_covid,  mu)
    tariff_metrics[name] = stress_metrics(w, ret_tariff, mu)
    print(f"  {name:<15} | COVID CVaR={covid_metrics[name]['realised_cvar']:.4f}"
          f"  | Tariff CVaR={tariff_metrics[name]['realised_cvar']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Figure A — Bar chart (CVaR / Drawdown / Sharpe)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Equal Weight": "#e07b54",
    "Gurobi":       "#2d6a9f",
    "NSGA-II":      "#5aaa5a",
    "MOEA/D":       "#b87333",
    "AGE-MOEA":     "#7b5ea7",
}

def _bar_group(ax, names, values, title, ylabel, invert=False):
    """Helper: grouped bar for one metric across portfolios."""
    x      = np.arange(len(names))
    colors = [COLORS[n] for n in names]
    bars   = ax.bar(x, values, color=colors, alpha=0.85, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5, rotation=15, ha="right")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Section 5.4 — Financial Robustness & Stress Testing\n"
             "COVID-19 (Feb–Apr 2020)  vs  2025 Trump Tariff Shock (Apr 2025)",
             fontsize=12)

names = list(portfolios.keys())
metric_spec = [
    ("realised_cvar", "Realised CVaR₉₅", "Loss (higher = worse)", False),
    ("max_drawdown",  "Max Drawdown",     "Drawdown (abs value)",  False),
    ("sharpe_ratio",  "Sharpe Ratio",     "Sharpe (higher = better)", True),
]

for col, (key, label, ylabel, higher_is_better) in enumerate(metric_spec):
    # COVID row
    vals_c = [abs(covid_metrics[n][key]) for n in names]
    _bar_group(axes[0, col], names, vals_c,
               f"COVID-19: {label}", ylabel)

    # Tariff row
    vals_t = [abs(tariff_metrics[n][key]) for n in names]
    _bar_group(axes[1, col], names, vals_t,
               f"Tariff Shock: {label}", ylabel)

axes[0, 0].set_ylabel("COVID-19\n" + metric_spec[0][2], fontsize=9)
axes[1, 0].set_ylabel("Tariff Shock\n" + metric_spec[0][2], fontsize=9)

plt.tight_layout()
plt.savefig("figures/stress_bar_chart.pdf", dpi=150, bbox_inches="tight")
plt.savefig("figures/stress_bar_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved figures/stress_bar_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Figure B — Cumulative return curves (the drawdown chart)
# ─────────────────────────────────────────────────────────────────────────────
def cumulative_return_curve(ret_window: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Cumulative portfolio value starting at 1.0."""
    port_ret   = ret_window.values @ w
    cum        = pd.Series((1 + port_ret).cumprod(), index=ret_window.index)
    return cum


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cumulative Portfolio Return During Stress Windows", fontsize=12)

for ax, (window_label, ret_window) in zip(
    axes,
    [("COVID-19 Shock (Feb–Apr 2020)", ret_covid),
     ("2025 Tariff Shock (Apr 2025)",  ret_tariff)]
):
    if ret_window.empty:
        ax.text(0.5, 0.5, "No data for this window",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(window_label, fontsize=11)
        continue

    for name, w in portfolios.items():
        curve = cumulative_return_curve(ret_window, w)
        ax.plot(curve.index, curve.values,
                label=name, color=COLORS[name], linewidth=2)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_title(window_label, fontsize=11)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Cumulative Return (rebased to 1.0)", fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig("figures/stress_cumulative.pdf", dpi=150, bbox_inches="tight")
plt.savefig("figures/stress_cumulative.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved figures/stress_cumulative.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Print summary table for the report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE — paste into report")
print("=" * 80)
print(f"\n{'Portfolio':<15} | {'COVID CVaR':>10} {'COVID DD':>10} {'COVID SR':>10}"
      f" | {'Tariff CVaR':>12} {'Tariff DD':>10} {'Tariff SR':>10}")
print("-" * 80)
for name in names:
    cm = covid_metrics[name]
    tm = tariff_metrics[name]
    cvar_c = cm['realised_cvar'] if cm['realised_cvar'] is not None else float('nan')
    dd_c   = cm['max_drawdown']  if cm['max_drawdown']  is not None else float('nan')
    sr_c   = cm['sharpe_ratio']  if cm['sharpe_ratio']  is not None else float('nan')
    cvar_t = tm['realised_cvar'] if tm['realised_cvar'] is not None else float('nan')
    dd_t   = tm['max_drawdown']  if tm['max_drawdown']  is not None else float('nan')
    sr_t   = tm['sharpe_ratio']  if tm['sharpe_ratio']  is not None else float('nan')
    print(f"{name:<15} | {cvar_c:>10.4f} {dd_c:>10.4f} {sr_c:>10.4f}"
          f" | {cvar_t:>12.4f} {dd_t:>10.4f} {sr_t:>10.4f}")

print("\nDone. Check figures/ folder for stress_bar_chart.png and stress_cumulative.png")
