"""
Rerun only the constraint-handling convergence experiment.

Use this script when `main_pipeline.py` has already produced
`pipeline_results.json`, but the constraint-handling HV-vs-time curve
needs to be regenerated. This avoids rerunning the full pipeline, which
can take many hours.

What it does:
    1. Loads cached prices via `fetch_prices()`.
    2. Rebuilds Repair, Penalty, and Decoder pymoo problems.
    3. Runs NSGA-II for each constraint-handling method.
    4. Records HV per generation and wall-clock time from the start of
       `pymoo_minimize()`.
    5. Backs up `pipeline_results.json`.
    6. Writes updated `constraint_handling_curves` into
       `pipeline_results.json`, so `generate_report_figures.py` can use
       the updated curves immediately.

This script still takes time because Repair is expensive, but it is much
shorter than rerunning the full main pipeline.
"""

import json
import os
import shutil
import time

import numpy as np

from fetch_universe import fetch_prices
from compute_return import compute_all
from nsga2_optimizer import (
    PortfolioProblem,
    PortfolioProblemPenalty,
    PortfolioProblemDecoder,
)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "pipeline_results.json")
BACKUP_PATH = os.path.join(PROJECT_ROOT, "pipeline_results_before_constraint_convergence.json")
N_GEN = 50
POP_SIZE = 500
SEED = 42


class HVCallback(Callback):
    """
    Record hypervolume and elapsed wall-clock time at each generation.

    The timer is set by `run_nsga2_with_curve()` before optimisation
    starts, so timestamps include initial population evaluation.
    """
    def __init__(self, hv_indicator):
        super().__init__()
        self.hv_ind = hv_indicator
        self.history = []
        self.timestamps = []
        self._t0 = None

    def notify(self, algorithm):
        if self._t0 is None:
            self._t0 = time.time()

        F = algorithm.pop.get("F").copy()
        F[:, 0] = -F[:, 0]
        try:
            hv_val = self.hv_ind.do(F)
        except Exception:
            hv_val = 0.0

        self.history.append(float(hv_val))
        self.timestamps.append(float(time.time() - self._t0))


def run_nsga2_with_curve(problem, sampling, crossover, mutation, hv_indicator,
                         n_gen=N_GEN, seed=SEED):
    """
    Run one NSGA-II convergence experiment and return curve diagnostics.

    Returns:
        Dictionary with `hv`, `cpu_time`, and `elapsed` keys.
    """
    callback = HVCallback(hv_indicator)
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    t0 = time.time()
    callback._t0 = t0
    pymoo_minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
        callback=callback,
    )
    elapsed = time.time() - t0

    return {
        "hv": [0.0] + callback.history,
        "cpu_time": [0.0] + callback.timestamps,
        "elapsed": float(elapsed),
    }


def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run main_pipeline.py first."
        )

    print(f"Loading {RESULTS_PATH}...")
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print("Loading prices and computing returns...")
    prices = fetch_prices()
    prices = prices.dropna(axis=1, how="all")
    tickers = list(prices.columns)
    ret, mu, _ = compute_all(prices)
    scenarios = ret.values

    print("Building constraint-handling problem instances...")
    problem_repair = PortfolioProblem(mu, scenarios, tickers)
    problem_penalty = PortfolioProblemPenalty(mu, scenarios, tickers)
    problem_decoder = PortfolioProblemDecoder(mu, scenarios, tickers)

    hv_ind = HV(ref_point=np.array([0.6, 0.10]))
    runs = {
        "Repair": (
            problem_repair,
            IntegerRandomSampling(),
            SBX(prob=0.9, eta=15, vtype=int),
            PM(eta=20, vtype=int),
        ),
        "Penalty": (
            problem_penalty,
            IntegerRandomSampling(),
            SBX(prob=0.9, eta=15, vtype=int),
            PM(eta=20, vtype=int),
        ),
        "Decoder": (
            problem_decoder,
            FloatRandomSampling(),
            SBX(prob=0.9, eta=15),
            PM(eta=20),
        ),
    }

    curves_gen = {}
    curves_time = {}
    elapsed_by_method = {}

    print(f"Running constraint-handling convergence only: {N_GEN} generations")
    for name, (problem, sampling, crossover, mutation) in runs.items():
        print(f"  {name}...")
        curve = run_nsga2_with_curve(
            problem, sampling, crossover, mutation, hv_ind
        )
        curves_gen[name] = curve["hv"]
        curves_time[name] = curve["cpu_time"]
        elapsed_by_method[name] = curve["elapsed"]
        print(
            f"    final HV={curve['hv'][-1]:.6f}, "
            f"curve time={curve['cpu_time'][-1]:.1f}s, "
            f"elapsed={curve['elapsed']:.1f}s"
        )

    results["constraint_handling_curves"] = {
        "generations": curves_gen,
        "cpu_time": curves_time,
    }
    results["constraint_handling_curve_rerun"] = {
        "n_gen": N_GEN,
        "pop_size": POP_SIZE,
        "seed": SEED,
        "elapsed_by_method": elapsed_by_method,
        "timer_note": (
            "Timestamps start before pymoo_minimize(), so they include "
            "initial population evaluation. A zero-HV baseline point is "
            "prepended so report plots start at t=0."
        ),
    }

    if os.path.exists(RESULTS_PATH):
        shutil.copy2(RESULTS_PATH, BACKUP_PATH)
        print(f"Backup saved to {BACKUP_PATH}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Updated {RESULTS_PATH}")
    print("Run generate_report_figures.py to regenerate the graph.")


if __name__ == "__main__":
    main()
