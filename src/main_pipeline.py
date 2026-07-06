"""
main_pipeline.py
=================
HST Project 1 -- Multi-Objective Portfolio Optimisation.
Group: Shetty Arya Dinesh, Buraporn Ruangchaem, Clarence Elvareta Kurniawan.

WHAT THIS SCRIPT DOES
----------------------
Runs the full study end to end, in order:

  1.  Data loading (cached prices -> log returns -> mu, Sigma)
  2.  Equal-weight baseline (CVaR, expected return)
  3.  Stress testing the equal-weight baseline (COVID-19, 2025 tariff shock)
  4.  Gurobi exact optimisation -- full epsilon-constraint Pareto front,
      plus "best portfolio" selection using REALISED Sharpe (not a
      Gaussian CVaR/1.65 proxy -- see the comment in Section 4 for why)
  5.  Gurobi scaling experiment (NP-hardness demonstration), using a
      STRATIFIED ticker sample at every universe size M (see Section 5
      comment for why this matters)
  6.  Constraint-handling comparison: Repair vs Penalty vs Decoder
  6.6 Constraint-handling: HV vs generation / HV vs CPU-time curves
  7.  Search-operator comparison: SBX vs Uniform Crossover vs Single-Point
  7.6 Search-operator: HV vs generation / HV vs CPU-time curves
  8.  Algorithm comparison: NSGA-II vs MOEA/D vs AGE-MOEA vs Gurobi
  9.  Statistical significance (15 independent runs, Mann-Whitney U test)
  10. Convergence curves (100 generations, NSGA-II / MOEA-D / AGE-MOEA)
  11. Portfolio weight / sector breakdown of the selected Gurobi portfolio
  12. Full stress-test comparison (equal-weight vs optimised portfolio)
  13. Export everything to pipeline_results.json

Run this once, then run generate_report_figures.py -- it reads
pipeline_results.json and produces every chart from real numbers, with
no manual copy-pasting of console output required.

WHY "final_sampling / final_crossover / final_mutation" EXIST
----------------------------------------------------------------
Sections 6, 7, 8, 9, and 10 all eventually need to run NSGA-II, MOEA/D,
or AGE-MOEA against `problem_final` -- whichever constraint-handling
strategy won Section 6 (Repair, Penalty, or Decoder). These three
strategies use DIFFERENT VARIABLE TYPES:
    - Repair / Penalty (PortfolioProblem / PortfolioProblemPenalty):
      an INTEGER genotype (a near-binary selection vector, vtype=int).
    - Decoder (PortfolioProblemDecoder): a CONTINUOUS genotype (a
      priority-score vector in [0,1], no vtype).

An earlier version of this pipeline hardcoded FloatRandomSampling() +
continuous SBX for "NSGA-II" in Section 10's convergence-curve
experiment, regardless of which constraint-handling method had won.
When Repair or Penalty won (integer problems), this silently fed
continuous, unbounded genotypes into a problem whose _evaluate()
calls z.astype(int) -- truncating every individual into a degenerate
selection that fails the problem's own cardinality guard clause,
producing a FIXED, single dominated point for the entire 100-generation
run (observed as: convergence_100gen["NSGA-II"] == [0.0]*100 in a real
run, while MOEA/D and AGE-MOEA in that same block showed real curves,
because only NSGA-II's config had been left stale).

THE FIX: `final_sampling`, `final_crossover`, and `final_mutation` are
computed ONCE, immediately after `best_ch` is known (end of Section 6),
based on best_ch's actual variable type, and reused for every algorithm
in Sections 8, 9, and 10. This makes the pipeline robust to whichever
constraint-handling method wins on a given run, rather than assuming
Repair (or Decoder) will always win.
"""

import json
import time
import numpy as np
from scipy.stats import mannwhitneyu

from fetch_universe import fetch_prices
from compute_return  import compute_all
from compute_cvar    import cvar, expected_return
from constraints     import is_feasible, repair
from stress_window   import covid_window, tariff_window, stress_metrics
from config import (ALL_TICKERS, K, SECTOR_MAP, BOND_ETFS, CRYPTOS,
                     BROAD_ETFS, SP500_EQUITIES, SECTOR_CAP, CRYPTO_CAP,
                     BOND_FLOOR)

from gurobi_optimizer import optimize_gurobi
from nsga2_optimizer   import (PortfolioProblem, PortfolioProblemPenalty,
                                PortfolioProblemDecoder)

from pymoo.algorithms.moo.nsga2    import NSGA2
from pymoo.algorithms.moo.moead    import MOEAD
from pymoo.algorithms.moo.age      import AGEMOEA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux  import UniformCrossover
from pymoo.operators.crossover.spx import SPX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize                import minimize as pymoo_minimize
from pymoo.indicators.hv           import HV
from pymoo.indicators.gd           import GD
from pymoo.indicators.igd          import IGD
from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.core.callback           import Callback


# =====================================================================
# CONFIG FOR THIS RUN -- tweak here, not scattered through the file
# =====================================================================
N_GEN_MAIN     = 50    # generations for all head-to-head comparisons
N_GEN_CONVERGE = 100   # generations for the dedicated convergence study
N_RUNS_STATS   = 15    # independent runs for statistical significance
                       # (reduced from 30 -> 15 for a faster targeted
                       # rerun; still valid for Mann-Whitney U, which
                       # only requires n >= ~8 for the normal
                       # approximation to be reasonable, at some cost
                       # to statistical power versus n=30)
POP_SIZE       = 500
SEED           = 42
RESULTS_PATH   = "pipeline_results.json"

# Scaling experiment universe sizes and Gurobi's per-run epsilon (single
# feasible point, not a full 20-point sweep -- see Section 5 docstring
# for why a full sweep would be a more rigorous NP-hardness demonstration)
SCALING_M_VALUES  = [20, 40, 60, 80, 100]
SCALING_EPSILON   = 0.05


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


class HVCallback(Callback):
    """
    pymoo Callback that records hypervolume AND elapsed wall-clock time
    at every generation. Used for all "HV vs generation" / "HV vs CPU
    time" plots in this study (Sections 6.6, 7.6, 10).
    """
    def __init__(self, hv_indicator):
        super().__init__()
        self.hv_ind = hv_indicator
        self.history = []      # HV per generation
        self.timestamps = []   # wall-clock seconds per generation, from
                                # the start of pymoo_minimize()
        self._t0 = None

    def notify(self, algorithm):
        if self._t0 is None:
            self._t0 = time.time()
        F = algorithm.pop.get("F").copy()
        F[:, 0] = -F[:, 0]  # un-negate return (pymoo minimises by default,
                             # but our objective 1 is annualised return,
                             # which we MAXIMISE -- see run_nsga2 docstring)
        try:
            hv_val = self.hv_ind.do(F)
        except Exception:
            # HV can occasionally fail on a degenerate/empty front
            # (e.g. every individual dominated or infeasible); treat
            # this generation's HV as 0 rather than crashing the run.
            hv_val = 0.0
        self.history.append(hv_val)
        self.timestamps.append(time.time() - self._t0)


def run_nsga2(problem, sampling, crossover, mutation, n_gen,
              callback=None, seed=SEED):
    """
    Single NSGA-II run. Returns (F_corrected, elapsed_seconds).

    F_corrected: the final population's objective matrix, with column 0
    (return) negated back to its true sign. pymoo's Problem classes in
    this study internally MINIMISE both objectives (so that everything
    is a standard multi-objective minimisation), meaning objective 1
    is stored internally as -return. We flip the sign back here so
    every F returned by this function reads directly as
    (annualised return, CVaR) in the natural, human-readable sense.
    """
    algo = NSGA2(
        pop_size=POP_SIZE,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    kwargs = {"seed": seed, "verbose": False}
    if callback is not None:
        kwargs["callback"] = callback

    t0 = time.time()
    if callback is not None and hasattr(callback, "_t0"):
        callback._t0 = t0
    res = pymoo_minimize(problem, algo, ("n_gen", n_gen), **kwargs)
    elapsed = time.time() - t0

    F = res.F.copy()
    F[:, 0] = -F[:, 0]
    return F, elapsed


def stratified_ticker_sample(m_target):
    """
    Build a representative sub-universe of size m_target, preserving
    (as closely as integer rounding allows) the same asset-class mix
    as the full M=100 universe: 80% equities, 6% bonds, 10% broad
    ETFs, 4% crypto.

    WHY THIS EXISTS: an earlier version of the Section 5 scaling
    experiment took tickers_full[:m_target] -- the FIRST m_target
    tickers in ALL_TICKERS's list order. Because config.py orders
    ALL_TICKERS as [equities..., bonds..., broad ETFs..., crypto...],
    every small-M scaling run (e.g. M=20) contained ONLY equities --
    zero bonds, zero ETFs, zero crypto. This meant the bond-floor and
    crypto-cap constraints were not exercised the same way at small M
    as they are at M=100, so the scaling comparison was not testing a
    like-for-like problem shape across M: solve-time differences could
    partly reflect "this M has fewer active constraints" rather than
    purely "this M has more binary variables."

    THE FIX: sample proportionally from each asset class at every M,
    so a shrunk M=20 universe still has (roughly) bonds, ETFs, and
    crypto in the same ratio as the real M=100 universe. Sampling is
    deterministic (first-N within each class, not random) so the
    experiment is exactly reproducible run to run.

    NOTE: this changes the scaling experiment's numbers relative to
    earlier runs of this pipeline (see report Limitations) -- the
    small-M problems are now genuinely harder (more constraint types
    active) and more representative, which is the point, but means
    solve times at M=20/40/60/80 should not be compared directly
    against numbers from a pre-fix run.
    """
    total = len(SP500_EQUITIES) + len(BOND_ETFS) + len(BROAD_ETFS) + len(CRYPTOS)
    frac_eq  = len(SP500_EQUITIES) / total
    frac_bd  = len(BOND_ETFS)      / total
    frac_etf = len(BROAD_ETFS)     / total
    frac_cr  = len(CRYPTOS)        / total

    n_eq  = max(1, round(m_target * frac_eq))
    n_bd  = max(1, round(m_target * frac_bd))
    n_etf = max(1, round(m_target * frac_etf))
    n_cr  = max(1, round(m_target * frac_cr))

    # Rounding can overshoot m_target by a couple of assets; trim from
    # the largest class (equities) first since it has the most slack.
    sampled = (SP500_EQUITIES[:n_eq] + BOND_ETFS[:n_bd]
               + BROAD_ETFS[:n_etf] + CRYPTOS[:n_cr])
    while len(sampled) > m_target:
        # remove the last equity added (keeps at least 1 of each other
        # class intact, since equities have the largest allocation)
        for i in range(len(sampled) - 1, -1, -1):
            if sampled[i] in SP500_EQUITIES:
                sampled.pop(i)
                break
        else:
            sampled.pop()  # fallback, shouldn't normally trigger
    while len(sampled) < m_target:
        # top up from equities (the largest pool) if rounding undershot
        remaining_eq = [t for t in SP500_EQUITIES if t not in sampled]
        if not remaining_eq:
            break
        sampled.append(remaining_eq[0])

    return sampled


def gurobi_scaling_experiment(mu_full, scenarios_full, tickers_full):
    """
    Runs Gurobi at a single epsilon point (SCALING_EPSILON) across
    SCALING_M_VALUES universe sizes, using a STRATIFIED ticker sample
    at each M (see stratified_ticker_sample docstring for why).

    Returns: {M: solve_time_seconds, ...}
    """
    results = {}
    for m_target in SCALING_M_VALUES:
        sub_tickers = stratified_ticker_sample(m_target)
        idx = [tickers_full.index(t) for t in sub_tickers]
        mu_sub = mu_full.values[idx] if hasattr(mu_full, "values") else mu_full[idx]
        scenarios_sub = scenarios_full[:, idx]

        n_eq  = sum(1 for t in sub_tickers if t in SP500_EQUITIES)
        n_bd  = sum(1 for t in sub_tickers if t in BOND_ETFS)
        n_etf = sum(1 for t in sub_tickers if t in BROAD_ETFS)
        n_cr  = sum(1 for t in sub_tickers if t in CRYPTOS)
        print(f"  M={m_target}: stratified sample = {n_eq} equities, "
              f"{n_bd} bonds, {n_etf} ETFs, {n_cr} crypto "
              f"({len(sub_tickers)} total)")

        t0 = time.time()
        fronts, _ = optimize_gurobi(mu_sub, scenarios_sub, sub_tickers,
                                     epsilon_values=[SCALING_EPSILON])
        elapsed = time.time() - t0
        results[m_target] = elapsed
        print(f"    -> solved in {elapsed:.2f}s "
              f"({len(fronts)} feasible point(s) found)")

    return results


def main():
    results = {}  # everything destined for pipeline_results.json

    # =================================================================
    section("1. DATA LOADING")
    # =================================================================
    prices = fetch_prices()
    prices = prices.dropna(axis=1, how="all")
    tickers = list(prices.columns)
    M = len(tickers)
    print(f"Assets      : {M}")
    print(f"Trading days: {len(prices)}")
    print(f"Date range  : {prices.index[0].date()} to {prices.index[-1].date()}")

    ret, mu, Sigma = compute_all(prices)
    scenarios = ret.values
    print(f"Mu range    : [{mu.min():.4f}, {mu.max():.4f}]")

    results["data"] = {
        "n_assets": M,
        "n_trading_days": len(prices),
        "date_start": str(prices.index[0].date()),
        "date_end": str(prices.index[-1].date()),
        "mu_min": float(mu.min()), "mu_max": float(mu.max()),
    }

    # =================================================================
    section("2. BASELINE EQUAL-WEIGHT PORTFOLIO")
    # =================================================================
    w_eq = np.ones(M) / M
    eq_cvar = cvar(w_eq, scenarios)
    eq_ret = expected_return(w_eq, mu.values)
    print(f"CVaR95    : {eq_cvar:.4f}")
    print(f"E[return] : {eq_ret:.4f}")
    results["baseline"] = {"cvar": float(eq_cvar), "expected_return": float(eq_ret)}

    # Sanity check: repair() + is_feasible() on a random K-asset draw.
    # This is a smoke test, not part of the main experiment -- it
    # exists to catch regressions in repair()/is_feasible() early,
    # since both are used throughout the rest of this script.
    rng = np.random.default_rng(SEED)
    z_test = np.zeros(M, dtype=int)
    z_test[rng.choice(M, K, replace=False)] = 1
    w_test = np.zeros(M)
    w_test[z_test == 1] = 1.0 / K
    z_test, w_test = repair(z_test, w_test, tickers)
    feasible, reasons = is_feasible(z_test, w_test, tickers)
    print(f"\nRepair-function sanity check feasible: {feasible}")
    if not feasible:
        for r in reasons:
            print(f"  \u2717 {r}")
        print("  (If this fails, double check repair()'s bond-floor "
              "top-up tolerance against is_feasible()'s strict check "
              "before trusting downstream Repair-strategy results.)")
    results["repair_sanity_check"] = {"feasible": bool(feasible), "reasons": reasons}

    # =================================================================
    section("3. STRESS TESTING -- EQUAL WEIGHT BASELINE")
    # =================================================================
    ret_covid = covid_window(ret)
    ret_tariff = tariff_window(ret)
    covid_eq = stress_metrics(w_eq, ret_covid, mu)
    tariff_eq = stress_metrics(w_eq, ret_tariff, mu)

    print(f"\n--- COVID-19 (Feb-Apr 2020) ---")
    print(f"Realised CVaR : {covid_eq['realised_cvar']:.4f}")
    print(f"Max Drawdown  : {covid_eq['max_drawdown']:.4f}  ({covid_eq['max_drawdown']*100:.2f}%)")
    print(f"Sharpe Ratio  : {covid_eq['sharpe_ratio']:.4f}")

    print(f"\n--- Trump Tariff Shock (2025) ---")
    print(f"Realised CVaR : {tariff_eq['realised_cvar']:.4f}")
    print(f"Max Drawdown  : {tariff_eq['max_drawdown']:.4f}  ({tariff_eq['max_drawdown']*100:.2f}%)")
    print(f"Sharpe Ratio  : {tariff_eq['sharpe_ratio']:.4f}")

    results["stress_equal_weight"] = {"covid": covid_eq, "tariff": tariff_eq}

    # =================================================================
    section("4. GUROBI EXACT OPTIMISATION -- PARETO FRONT")
    # =================================================================
    eq_cvar = cvar(np.ones(M) / M, scenarios)
    epsilons = np.linspace(eq_cvar * 0.5, eq_cvar * 3.0, 20)
    print(f"Epsilon range: [{epsilons[0]:.4f}, {epsilons[-1]:.4f}]")
    print(f"Sweeping {len(epsilons)} epsilon points...")

    t0 = time.time()
    fronts, _ = optimize_gurobi(mu, scenarios, tickers, epsilon_values=epsilons)
    gurobi_time = time.time() - t0

    print(f"\nGurobi finished in  : {gurobi_time:.2f} seconds")
    print(f"Pareto points found : {len(fronts)}")

    print(f"\n{'E[Return]':<14} {'CVaR95':<12} {'# Assets'}")
    print("-" * 38)
    for pt in fronts:
        print(f"{pt['ret']:<14.4f} {pt['cvar']:<12.4f} {int(sum(pt['z']))}")

    # --- Portfolio selection: REALISED Sharpe, not a Gaussian proxy. ---
    # An earlier version of this pipeline selected the "best" portfolio
    # by maximising an approximate Sharpe ratio computed as
    #     (ret - rf) / (cvar / 1.65)
    # using the Gaussian identity CVaR_0.95 ~= 1.65*sigma to back-solve
    # an implied standard deviation from CVaR alone. This produced an
    # implausibly high Sharpe (~15) because the identity assumes
    # normally distributed returns; the realised distribution --
    # especially with cryptocurrency exposure -- has heavier tails than
    # the Gaussian, so CVaR is elevated relative to what the identity
    # predicts, and back-solving sigma from an elevated CVaR under a
    # false normality assumption UNDERSTATES sigma, inflating the
    # resulting Sharpe estimate. This is a modelling artefact of the
    # selection step, not a real property of the portfolio.
    #
    # Fix: compute the REALISED Sharpe ratio directly from the
    # historical scenario returns for each candidate weight vector,
    # exactly as stress_metrics() already does elsewhere in this
    # pipeline (annualised return / annualised volatility of the
    # actual daily portfolio returns).
    rf = 0.02
    best_sharpe = -np.inf
    best_portfolio = None
    for pt in fronts:
        port_ret_series = scenarios @ pt["w"]
        ann_ret = port_ret_series.mean() * 252
        ann_vol = port_ret_series.std() * (252 ** 0.5)
        realised_sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else -np.inf
        pt["realised_sharpe"] = realised_sharpe
        if realised_sharpe > best_sharpe:
            best_sharpe = realised_sharpe
            best_portfolio = pt

    w_best = best_portfolio["w"]
    selected = [tickers[i] for i, z in enumerate(best_portfolio["z"]) if z == 1]

    print(f"\n--- BEST PORTFOLIO (Max REALISED Sharpe) ---")
    print(f"Expected Return  : {best_portfolio['ret']:.4f}  ({best_portfolio['ret']*100:.2f}%)")
    print(f"CVaR95           : {best_portfolio['cvar']:.4f}  ({best_portfolio['cvar']*100:.2f}%)")
    print(f"Realised Sharpe  : {best_sharpe:.4f}")
    print(f"# Assets         : {len(selected)}")
    print(f"Tickers          : {', '.join(selected)}")
    print(f"\nWeight allocation:")
    weight_table = {}
    for i, t in enumerate(tickers):
        if best_portfolio["z"][i] == 1:
            print(f"  {t:<12} {w_best[i]:.4f}  ({w_best[i]*100:.2f}%)")
            weight_table[t] = float(w_best[i])

    gurobi_F = np.array([[pt["ret"], pt["cvar"]] for pt in fronts])

    results["gurobi_pareto_front"] = {
        "epsilons": epsilons.tolist(),
        "returns": [float(pt["ret"]) for pt in fronts],
        "cvars": [float(pt["cvar"]) for pt in fronts],
        "n_assets": [int(sum(pt["z"])) for pt in fronts],
        "solve_time_sec": gurobi_time,
    }
    results["best_portfolio"] = {
        "expected_return": float(best_portfolio["ret"]),
        "cvar": float(best_portfolio["cvar"]),
        "realised_sharpe": float(best_sharpe),
        "n_assets": len(selected),
        "tickers": selected,
        "weights": weight_table,
    }

    section("5. GUROBI SCALING EXPERIMENT -- NP-HARDNESS PROOF")
    # =================================================================

    from gurobi_optimizer import scaling_experiment_proper

    print("Running scaling experiment...")
    print("Each problem size solves a complete 20-point Pareto front.")

    t0 = time.time()

    scaling_results = scaling_experiment_proper(
        mu,
        scenarios,
        tickers,
        n_epsilons=20,
        time_limit=300
    )

    total_scaling_time = time.time() - t0

    print("\n--- SCALING RESULTS ---")
    print(f"{'M':<8} {'Time(s)':<12} {'Relative'}")
    print("-"*35)

    base = scaling_results[20]

    for m in sorted(scaling_results.keys()):
        print(f"{m:<8} {scaling_results[m]:<12.2f} {scaling_results[m]/base:.2f}x")

    print(f"\nTotal experiment time: {total_scaling_time:.2f} seconds")

    results["scaling"] = {str(m): float(t) for m, t in scaling_results.items()}
    results["scaling_config"] = {
        "pareto_points": 20,
        "total_experiment_time": float(total_scaling_time)
    }
    # =================================================================
    section("6. CONSTRAINT HANDLING: REPAIR vs PENALTY vs DECODER")
    # =================================================================
    hv_ind = HV(ref_point=np.array([0.6, 0.10]))
    gd_ind = GD(gurobi_F)
    igd_ind = IGD(gurobi_F)

    problem_repair = PortfolioProblem(mu, scenarios, tickers)
    problem_penalty = PortfolioProblemPenalty(mu, scenarios, tickers)
    problem_decoder = PortfolioProblemDecoder(mu, scenarios, tickers)

    constraint_results = {}
    ch_runs = {
        "Repair":  (problem_repair,  IntegerRandomSampling(),
                    SBX(prob=0.9, eta=15, vtype=int), PM(eta=20, vtype=int)),
        "Penalty": (problem_penalty, IntegerRandomSampling(),
                    SBX(prob=0.9, eta=15, vtype=int), PM(eta=20, vtype=int)),
        "Decoder": (problem_decoder, FloatRandomSampling(),
                    SBX(prob=0.9, eta=15), PM(eta=20)),
    }

    for name, (problem, sampling, crossover, mutation) in ch_runs.items():
        print(f"Running NSGA-II with {name}...")
        F, elapsed = run_nsga2(problem, sampling, crossover, mutation, N_GEN_MAIN)
        constraint_results[name] = {
            "HV": hv_ind.do(F), "GD": gd_ind.do(F), "IGD": igd_ind.do(F),
            "time": elapsed, "F": F,
        }
        print(f"  Done in {elapsed:.1f}s -- HV={constraint_results[name]['HV']:.6f}")

    print(f"\n--- CONSTRAINT HANDLING RESULTS ---")
    print(f"{'Method':<12} {'HV':>12} {'GD':>12} {'IGD':>12} {'Time(s)':>10}")
    print("-" * 60)
    for name, m in constraint_results.items():
        print(f"{name:<12} {m['HV']:>12.6f} {m['GD']:>12.6f} {m['IGD']:>12.6f} {m['time']:>10.1f}")

    best_ch = max(constraint_results, key=lambda k: constraint_results[k]["HV"])
    print(f"\nWinner: {best_ch}")

    problem_final = {"Repair": problem_repair, "Penalty": problem_penalty,
                      "Decoder": problem_decoder}[best_ch]
    nsga_F = constraint_results[best_ch]["F"]

    # --- THE KEY FIX (see module docstring for full explanation) -------
    # Define the operator set ONCE here, based on best_ch's actual
    # variable type, and reuse it for every algorithm in Sections 8, 9,
    # and 10 below. Repair and Penalty use an integer genotype;
    # Decoder uses a continuous genotype. Using the wrong sampling/
    # crossover/mutation type for a given problem silently corrupts
    # that algorithm's results (observed previously as NSGA-II's
    # 100-generation convergence curve being exactly 0.0 throughout,
    # when Repair had won but Section 10 still hardcoded continuous
    # operators left over from an assumption that Decoder would win).
    is_integer_problem = best_ch in ("Repair", "Penalty")
    if is_integer_problem:
        final_sampling  = IntegerRandomSampling()
        final_crossover = SBX(prob=0.9, eta=15, vtype=int)
        final_mutation  = PM(eta=20, vtype=int)
    else:  # Decoder: continuous genotype
        final_sampling  = FloatRandomSampling()
        final_crossover = SBX(prob=0.9, eta=15)
        final_mutation  = PM(eta=20)
    print(f"(Sections 8/9/10 will use "
          f"{'integer' if is_integer_problem else 'continuous'} "
          f"operators to match {best_ch}'s genotype.)")

    results["constraint_handling"] = {
        name: {"HV": float(m["HV"]), "GD": float(m["GD"]), "IGD": float(m["IGD"]),
               "time": float(m["time"])}
        for name, m in constraint_results.items()
    }
    results["constraint_handling"]["winner"] = best_ch

    # =================================================================
    section("6.6 CONSTRAINT HANDLING: HV vs GENERATION / HV vs CPU TIME")
    # =================================================================
    # NOTE: Repair's per-generation repair() projection is genuinely
    # expensive at population 500; this step can take significantly
    # longer than Penalty or Decoder purely because of that cost, not
    # because of any bug.
    ch_convergence = {}
    ch_time_curve = {}
    for name, (problem, sampling, crossover, mutation) in ch_runs.items():
        print(f"Running {name} for {N_GEN_MAIN} generations (tracking HV + time per gen)...")
        cb = HVCallback(hv_ind)
        _, elapsed = run_nsga2(problem, sampling, crossover, mutation, N_GEN_MAIN, callback=cb)
        ch_convergence[name] = cb.history
        ch_time_curve[name] = cb.timestamps
        print(f"  Done -- final HV={cb.history[-1]:.6f}, total time={cb.timestamps[-1]:.1f}s")

    results["constraint_handling_curves"] = {
        "generations": {name: hv_list for name, hv_list in ch_convergence.items()},
        "cpu_time": {name: t_list for name, t_list in ch_time_curve.items()},
    }

    # =================================================================
    section("7. SEARCH OPERATOR VARIATION: SBX vs UNIFORM vs SINGLE-POINT")
    # =================================================================
    # This comparison is deliberately always run under Decoder
    # (problem_decoder, FloatRandomSampling), independent of which
    # constraint-handling method won Section 6 -- the operator
    # comparison's purpose is to isolate crossover-operator effects on
    # a fixed, continuous search space, not to be re-run under whatever
    # won Section 6. If Section 6's winner is later used for Section
    # 8/9/10 algorithm comparisons, that is a SEPARATE, deliberate
    # choice (compare algorithms under the best-performing constraint
    # handling), not an inconsistency with this section.
    operator_results = {}
    op_runs = {
        "SBX + PM":          SBX(prob=0.9, eta=15),
        "Uniform Crossover": UniformCrossover(prob=0.9),
        "Single-Point":      SPX(prob=0.9),
    }

    for name, crossover in op_runs.items():
        print(f"Running {name}...")
        F, elapsed = run_nsga2(problem_decoder, FloatRandomSampling(), crossover,
                                PM(eta=20), N_GEN_MAIN)
        operator_results[name] = {
            "HV": hv_ind.do(F), "GD": gd_ind.do(F), "IGD": igd_ind.do(F),
            "time": elapsed, "F": F,
        }
        print(f"  Done in {elapsed:.1f}s -- HV={operator_results[name]['HV']:.6f}")

    print(f"\n--- SEARCH OPERATOR RESULTS ---")
    print(f"{'Operator':<25} {'HV':>12} {'GD':>12} {'IGD':>12} {'Time(s)':>10}")
    print("-" * 73)
    for name, m in operator_results.items():
        print(f"{name:<25} {m['HV']:>12.6f} {m['GD']:>12.6f} {m['IGD']:>12.6f} {m['time']:>10.1f}")

    best_op = max(operator_results, key=lambda k: operator_results[k]["HV"])
    print(f"\nWinner: {best_op}")

    results["search_operators"] = {
        name: {"HV": float(m["HV"]), "GD": float(m["GD"]), "IGD": float(m["IGD"]),
               "time": float(m["time"])}
        for name, m in operator_results.items()
    }
    results["search_operators"]["winner"] = best_op

    # =================================================================
    section("7.6 SEARCH OPERATORS: HV vs GENERATION / HV vs CPU TIME")
    # =================================================================
    op_convergence = {}
    op_time_curve = {}
    for name, crossover in op_runs.items():
        print(f"Running {name} for {N_GEN_MAIN} generations (tracking HV + time per gen)...")
        cb = HVCallback(hv_ind)
        _, elapsed = run_nsga2(problem_decoder, FloatRandomSampling(), crossover,
                                PM(eta=20), N_GEN_MAIN, callback=cb)
        op_convergence[name] = cb.history
        op_time_curve[name] = cb.timestamps
        print(f"  Done -- final HV={cb.history[-1]:.6f}, total time={cb.timestamps[-1]:.1f}s")

    results["search_operator_curves"] = {
        "generations": {name: hv_list for name, hv_list in op_convergence.items()},
        "cpu_time": {name: t_list for name, t_list in op_time_curve.items()},
    }

    # =================================================================
    section("8. ALGORITHM COMPARISON: NSGA-II vs MOEA/D vs AGE-MOEA")
    # =================================================================
    # All three algorithms below run on problem_final (Section 6's
    # winning constraint-handling strategy) using final_sampling /
    # final_crossover / final_mutation, defined once in Section 6 to
    # match problem_final's actual genotype -- see the "THE KEY FIX"
    # comment there for why this matters.

    print(f"Running NSGA-II (fresh run on problem_final = {best_ch})...")
    nsga_F, nsga_time = run_nsga2(problem_final, final_sampling, final_crossover,
                                   final_mutation, N_GEN_MAIN, seed=SEED)
    print(f"  Done in {nsga_time:.1f}s -- HV={hv_ind.do(nsga_F):.6f}")

    print("Running MOEA/D...")
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    t0 = time.time()
    res_moead = pymoo_minimize(
        problem_final,
        MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.9,
              sampling=final_sampling, crossover=final_crossover,
              mutation=final_mutation),
        ("n_gen", N_GEN_MAIN), seed=SEED, verbose=False
    )
    moead_time = time.time() - t0
    moead_F = res_moead.F.copy(); moead_F[:, 0] = -moead_F[:, 0]
    print(f"  Done in {moead_time:.1f}s")

    print("Running AGE-MOEA...")
    t0 = time.time()
    res_age = pymoo_minimize(
        problem_final,
        AGEMOEA(pop_size=POP_SIZE, sampling=final_sampling,
                crossover=final_crossover, mutation=final_mutation,
                eliminate_duplicates=True),
        ("n_gen", N_GEN_MAIN), seed=SEED, verbose=False
    )
    age_time = time.time() - t0
    age_F = res_age.F.copy(); age_F[:, 0] = -age_F[:, 0]
    print(f"  Done in {age_time:.1f}s")

    algo_metrics = {
        "NSGA-II":  {"HV": hv_ind.do(nsga_F),  "GD": gd_ind.do(nsga_F),  "IGD": igd_ind.do(nsga_F),  "time": nsga_time},
        "MOEA/D":   {"HV": hv_ind.do(moead_F), "GD": gd_ind.do(moead_F), "IGD": igd_ind.do(moead_F), "time": moead_time},
        "AGE-MOEA": {"HV": hv_ind.do(age_F),   "GD": gd_ind.do(age_F),   "IGD": igd_ind.do(age_F),   "time": age_time},
        "Gurobi":   {"HV": hv_ind.do(gurobi_F), "GD": 0.0, "IGD": 0.0, "time": gurobi_time},
    }

    print(f"\n--- ALGORITHM COMPARISON RESULTS ---")
    print(f"{'Algorithm':<12} {'HV':>12} {'GD':>12} {'IGD':>12} {'Time(s)':>10}")
    print("-" * 60)
    for name, m in algo_metrics.items():
        print(f"{name:<12} {m['HV']:>12.6f} {m['GD']:>12.6f} {m['IGD']:>12.6f} {m['time']:>10.1f}")

    results["algorithm_comparison"] = {
        name: {"HV": float(m["HV"]), "GD": float(m["GD"]), "IGD": float(m["IGD"]),
               "time": float(m["time"])}
        for name, m in algo_metrics.items()
    }

    # =================================================================
    section("9. STATISTICAL SIGNIFICANCE -- 15 RUNS + WILCOXON TEST")
    # =================================================================
    # All three algorithms use final_sampling/final_crossover/
    # final_mutation, consistent with Section 8 above (see Section 6's
    # "THE KEY FIX" comment).
    hv_runs = {"NSGA-II": [], "MOEA/D": [], "AGE-MOEA": []}
    print(f"Running {N_RUNS_STATS} independent runs per algorithm...")

    for run_idx in range(N_RUNS_STATS):
        seed = run_idx * 7

        F, _ = run_nsga2(problem_final, final_sampling, final_crossover,
                          final_mutation, N_GEN_MAIN, seed=seed)
        hv_runs["NSGA-II"].append(hv_ind.do(F))

        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        res = pymoo_minimize(
            problem_final,
            MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.9,
                  sampling=final_sampling, crossover=final_crossover,
                  mutation=final_mutation),
            ("n_gen", N_GEN_MAIN), seed=seed, verbose=False
        )
        F = res.F.copy(); F[:, 0] = -F[:, 0]
        hv_runs["MOEA/D"].append(hv_ind.do(F))

        res = pymoo_minimize(
            problem_final,
            AGEMOEA(pop_size=POP_SIZE, sampling=final_sampling,
                    crossover=final_crossover, mutation=final_mutation,
                    eliminate_duplicates=True),
            ("n_gen", N_GEN_MAIN), seed=seed, verbose=False
        )
        F = res.F.copy(); F[:, 0] = -F[:, 0]
        hv_runs["AGE-MOEA"].append(hv_ind.do(F))

        print(f"  Run {run_idx+1:2d}/{N_RUNS_STATS} -- "
              f"NSGA-II: {hv_runs['NSGA-II'][-1]:.4f}  "
              f"MOEA/D: {hv_runs['MOEA/D'][-1]:.4f}  "
              f"AGE-MOEA: {hv_runs['AGE-MOEA'][-1]:.4f}")

    hv_stats = {}
    for algo, hvs in hv_runs.items():
        arr = np.array(hvs)
        hv_stats[algo] = {"mean": float(arr.mean()), "std": float(arr.std()),
                           "min": float(arr.min()), "max": float(arr.max()),
                           "median": float(np.median(arr)), "values": arr.tolist()}

    print(f"\n--- HV STATISTICS ACROSS {N_RUNS_STATS} RUNS ---")
    print(f"{'Algorithm':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print("-" * 60)
    for algo, s in hv_stats.items():
        print(f"{algo:<12} {s['mean']:>10.6f} {s['std']:>10.6f} {s['min']:>10.6f} {s['max']:>10.6f} {s['median']:>10.6f}")

    pairs = [("NSGA-II", "MOEA/D"), ("NSGA-II", "AGE-MOEA"), ("MOEA/D", "AGE-MOEA")]
    wilcoxon_results = {}
    print(f"\n--- WILCOXON / MANN-WHITNEY U TESTS ---")
    for a, b in pairs:
        stat, p = mannwhitneyu(hv_stats[a]["values"], hv_stats[b]["values"], alternative="two-sided")
        sig = "YES" if p < 0.05 else "NO"
        label = f"{a} vs {b}"
        wilcoxon_results[label] = {"stat": float(stat), "p": float(p), "significant": sig}
        print(f"{label:<30} p={p:.6f}  significant={sig}")

    results["hv_stats_30_runs"] = hv_stats  # key name kept for backward
                                             # compatibility with
                                             # generate_report_figures.py,
                                             # even though N_RUNS_STATS
                                             # is now 15, not 30
    results["wilcoxon"] = wilcoxon_results
    results["n_runs_stats"] = N_RUNS_STATS

    # =================================================================
    section("10. CONVERGENCE CURVES (100 generations)")
    # =================================================================
    # All three algorithms use final_sampling/final_crossover/
    # final_mutation, consistent with Sections 8 and 9 above. This is
    # the exact fix for the bug described in the module docstring.
    convergence = {}
    print(f"Running convergence experiment ({N_GEN_CONVERGE} generations)...")

    for algo_name, algo_obj in [
        ("NSGA-II", NSGA2(pop_size=POP_SIZE, sampling=final_sampling,
                           crossover=final_crossover, mutation=final_mutation,
                           eliminate_duplicates=True)),
        ("MOEA/D",  MOEAD(ref_dirs=get_reference_directions("das-dennis", 2, n_partitions=12),
                           n_neighbors=15, prob_neighbor_mating=0.9,
                           sampling=final_sampling,
                           crossover=final_crossover,
                           mutation=final_mutation)),
        ("AGE-MOEA", AGEMOEA(pop_size=POP_SIZE, sampling=final_sampling,
                              crossover=final_crossover, mutation=final_mutation,
                              eliminate_duplicates=True)),
    ]:
        print(f"  {algo_name}...")
        cb = HVCallback(hv_ind)
        pymoo_minimize(problem_final, algo_obj, ("n_gen", N_GEN_CONVERGE),
                        seed=SEED, verbose=False, callback=cb)
        convergence[algo_name] = cb.history
        print(f"    Final HV at generation {N_GEN_CONVERGE}: {cb.history[-1]:.6f}")

    print(f"\nFinal HV at generation {N_GEN_CONVERGE}:")
    for name, hv_list in convergence.items():
        print(f"  {name:<10}: {hv_list[-1]:.6f}")

    results["convergence_100gen"] = convergence

    # =================================================================
    section("11. PORTFOLIO WEIGHT / SECTOR ANALYSIS")
    # =================================================================
    print(f"\n--- ASSET ALLOCATION ---")
    print(f"{'Ticker':<12} {'Asset Class':<22} {'Sector':<30} {'Weight':>8} {'Weight%':>9}")
    print("-" * 85)

    sector_weights = {}
    allocation_table = []
    for i, t in enumerate(tickers):
        if best_portfolio["z"][i] == 1:
            w_i = w_best[i]
            sector = SECTOR_MAP.get(t, "Unknown")
            if t in BOND_ETFS:
                asset_class = "Bond ETF"
            elif t in CRYPTOS:
                asset_class = "Cryptocurrency"
            elif t in BROAD_ETFS:
                asset_class = "Broad ETF"
            else:
                asset_class = "S&P 500 Equity"
            print(f"{t:<12} {asset_class:<22} {sector:<30} {w_i:>8.4f} {w_i*100:>8.2f}%")
            sector_weights[sector] = sector_weights.get(sector, 0) + w_i
            allocation_table.append({"ticker": t, "asset_class": asset_class,
                                      "sector": sector, "weight": float(w_i)})

    print(f"\n--- SECTOR BREAKDOWN ---")
    sector_breakdown = {}
    for sector, w_s in sorted(sector_weights.items(), key=lambda x: -x[1]):
        sector_breakdown[sector] = float(w_s)
        print(f"  {sector:<30} {w_s*100:>8.2f}%")

    bond_w = sum(w_best[i] for i, t in enumerate(tickers) if t in BOND_ETFS and best_portfolio["z"][i] == 1)
    crypto_w = sum(w_best[i] for i, t in enumerate(tickers) if t in CRYPTOS and best_portfolio["z"][i] == 1)
    n_assets = int(sum(best_portfolio["z"]))
    w_sum = sum(w_best[i] for i in range(M) if best_portfolio["z"][i] == 1)
    max_sec_w = max(sector_weights.values()) if sector_weights else 0

    print(f"\n--- CONSTRAINT VERIFICATION ---")
    print(f"  Budget (sum=1)    : {w_sum:.4f}")
    print(f"  Cardinality (<=15): {n_assets}")
    print(f"  Bond floor (>=10%): {bond_w:.4f}")
    print(f"  Crypto cap (<=20%): {crypto_w:.4f}")
    print(f"  Sector cap (<=40%): {max_sec_w:.4f}")

    results["portfolio_allocation"] = allocation_table
    results["sector_breakdown"] = sector_breakdown
    results["constraint_verification"] = {
        "budget_sum": float(w_sum), "cardinality": n_assets,
        "bond_weight": float(bond_w), "crypto_weight": float(crypto_w),
        "max_sector_weight": float(max_sec_w),
    }

    # =================================================================
    section("12. FULL STRESS TEST COMPARISON")
    # =================================================================
    covid_opt = stress_metrics(w_best, ret_covid, mu)
    tariff_opt = stress_metrics(w_best, ret_tariff, mu)

    print(f"\n--- STRESS TEST: EQUAL-WEIGHT vs OPTIMISED ---")
    print(f"{'Metric':<18} {'COVID EW':>10} {'COVID Opt':>11} {'Tariff EW':>11} {'Tariff Opt':>12}")
    print("-" * 66)
    for label, key in [("Realised CVaR", "realised_cvar"),
                        ("Max Drawdown", "max_drawdown"),
                        ("Sharpe Ratio", "sharpe_ratio")]:
        print(f"{label:<18} {covid_eq[key]:>10.4f} {covid_opt[key]:>11.4f} "
              f"{tariff_eq[key]:>11.4f} {tariff_opt[key]:>12.4f}")

    results["stress_optimised"] = {"covid": covid_opt, "tariff": tariff_opt}

    # =================================================================
    section("13. SAVING RESULTS")
    # =================================================================
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_PATH}")
    print("Run generate_report_figures.py next to produce every chart from this file.")


if __name__ == "__main__":
    main()
