# rerun_stats.py
import numpy as np
import time
import json
from scipy.stats import mannwhitneyu

from fetch_universe  import fetch_prices
from compute_return  import compute_all
from config          import BOND_ETFS, CRYPTOS
from nsga2_optimizer import PortfolioProblem, PortfolioProblemDecoder, PortfolioProblemPenalty

from pymoo.algorithms.moo.nsga2    import NSGA2
from pymoo.algorithms.moo.moead    import MOEAD
from pymoo.algorithms.moo.age      import AGEMOEA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize                import minimize as pymoo_minimize
from pymoo.indicators.hv           import HV
from pymoo.indicators.gd           import GD
from pymoo.indicators.igd          import IGD
from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.core.callback           import Callback

# ── Step 1: reload data from cache ────────────────────────────────────────
print("Loading cached prices...")
prices    = fetch_prices()
prices    = prices.dropna(axis=1, how="all")
tickers   = list(prices.columns)
M         = len(tickers)
ret, mu, Sigma = compute_all(prices)
scenarios = ret.values
print(f"\u2713 {M} assets, {len(prices)} days")

# ── Step 2: reload Gurobi front from saved JSON ───────────────────────────
print("Loading pipeline_results.json...")
with open("pipeline_results.json") as f:
    saved = json.load(f)

returns  = saved["gurobi_pareto_front"]["returns"]
cvars    = saved["gurobi_pareto_front"]["cvars"]
gurobi_F = np.array([[r, c] for r, c in zip(returns, cvars)])
print(f"\u2713 Gurobi front: {gurobi_F.shape}")

# ── Step 3: rebuild problems ───────────────────────────────────────────────
print("Building problem instances...")
best_ch = saved["constraint_handling"]["winner"]
print(f"\u2713 Constraint handling winner: {best_ch}")

# NOTE: final_sampling / final_crossover / final_mutation are chosen ONCE
# here based on best_ch's actual variable type, and MUST be reused for
# ALL THREE algorithms (NSGA-II, MOEA/D, AGE-MOEA) everywhere below --
# in both the N_RUNS statistical loop and the convergence algo_configs.
# A previous version of this script hardcoded IntegerRandomSampling() +
# integer-vtype SBX/PM for MOEA/D and AGE-MOEA unconditionally, which only
# happened to work when best_ch == "Repair" or "Penalty" (both integer
# problems). If best_ch == "Decoder" (continuous genotype, xl=0.0/xu=1.0,
# no vtype=int), that hardcoding silently feeds integer-sampled populations
# into a continuous problem -- the same class of bug that previously
# collapsed NSGA-II's HV to exactly 0.0 in Section 8/9. Fixed by defining
# the operators once, from best_ch, and reusing them everywhere.
if best_ch == "Repair":
    problem_final   = PortfolioProblem(mu, scenarios, tickers)
    final_sampling  = IntegerRandomSampling()
    final_crossover = SBX(prob=0.9, eta=15, vtype=int)
    final_mutation  = PM(eta=20, vtype=int)
elif best_ch == "Decoder":
    problem_final   = PortfolioProblemDecoder(mu, scenarios, tickers)
    final_sampling  = FloatRandomSampling()
    final_crossover = SBX(prob=0.9, eta=15)
    final_mutation  = PM(eta=20)
else:  # Penalty
    problem_final   = PortfolioProblemPenalty(mu, scenarios, tickers)
    final_sampling  = IntegerRandomSampling()
    final_crossover = SBX(prob=0.9, eta=15, vtype=int)
    final_mutation  = PM(eta=20, vtype=int)

# ── Step 4: indicators ────────────────────────────────────────────────────
hv_ind  = HV(ref_point=np.array([0.6, 0.10]))
gd_ind  = GD(gurobi_F)
igd_ind = IGD(gurobi_F)

# =====================================================================
# SECTION 7.5 — N_RUNS independent runs + Wilcoxon
# Change N_RUNS to 10 or 25 depending on your time budget:
#   10 runs ≈ 1.5 hours
#   25 runs ≈ 3.5 hours
#   30 runs ≈ 4+ hours
# =====================================================================
N_RUNS     = 10   # ← change to 25 if you have more time
N_GEN_STAT = 50   # generations per statistical run (keep 50 for speed)

print("\n" + "="*60)
print(f"  SECTION 7.5 — {N_RUNS} RUNS + WILCOXON (50 gens each)")
print("="*60)
print(f"Estimated time: ~{N_RUNS * 3 * 35 / 60:.0f} minutes\n")

hv_runs   = {"NSGA-II": [], "MOEA/D": [], "AGE-MOEA": []}
time_runs = {"NSGA-II": [], "MOEA/D": [], "AGE-MOEA": []}

for run_idx in range(N_RUNS): 
    seed = run_idx * 7

    # NSGA-II
    t0  = time.time()
    res = pymoo_minimize(
        problem_final,
        NSGA2(
            pop_size=200,
            sampling=final_sampling,
            crossover=final_crossover,
            mutation=final_mutation,
            eliminate_duplicates=True
        ),
        ("n_gen", N_GEN_STAT), seed=seed, verbose=False
    )
    time_runs["NSGA-II"].append(time.time() - t0)
    F = res.F.copy(); F[:, 0] = -F[:, 0]
    hv_runs["NSGA-II"].append(hv_ind.do(F))

    # MOEA/D -- now uses final_sampling/final_crossover/final_mutation,
    # matching problem_final's actual variable type (fix, see note above)
    t0       = time.time()
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    res      = pymoo_minimize(
        problem_final,
        MOEAD(
            ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.9,
            sampling=final_sampling,
            crossover=final_crossover,
            mutation=final_mutation
        ),
        ("n_gen", N_GEN_STAT), seed=seed, verbose=False
    )
    time_runs["MOEA/D"].append(time.time() - t0)
    F = res.F.copy(); F[:, 0] = -F[:, 0]
    hv_runs["MOEA/D"].append(hv_ind.do(F))

    # AGE-MOEA -- same fix
    t0  = time.time()
    res = pymoo_minimize(
        problem_final,
        AGEMOEA(
            pop_size=500,
            sampling=final_sampling,
            crossover=final_crossover,
            mutation=final_mutation,
            eliminate_duplicates=True
        ),
        ("n_gen", N_GEN_STAT), seed=seed, verbose=False
    )
    time_runs["AGE-MOEA"].append(time.time() - t0)
    F = res.F.copy(); F[:, 0] = -F[:, 0]
    hv_runs["AGE-MOEA"].append(hv_ind.do(F))

    print(f"  Run {run_idx+1:2d}/{N_RUNS} — "
          f"NSGA-II: {hv_runs['NSGA-II'][-1]:.4f}  "
          f"MOEA/D: {hv_runs['MOEA/D'][-1]:.4f}  "
          f"AGE-MOEA: {hv_runs['AGE-MOEA'][-1]:.4f}")

# statistics
print(f"\n--- HV STATISTICS (copy into report) ---")
print(f"{'Algorithm':<12} {'Mean':>10} {'Std':>10} "
      f"{'Min':>10} {'Max':>10} {'Median':>10}")
print("-" * 60)
hv_stats = {}
for algo, hvs in hv_runs.items():
    arr = np.array(hvs)
    hv_stats[algo] = {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "median": float(np.median(arr)),
        "values": arr.tolist(),
    }
    print(f"{algo:<12} {arr.mean():>10.6f} {arr.std():>10.6f} "
          f"{arr.min():>10.6f} {arr.max():>10.6f} {np.median(arr):>10.6f}")

# Wilcoxon
print(f"\n--- WILCOXON RANK-SUM TESTS ---")
print(f"Note: using {N_RUNS} runs. Wilcoxon is valid for n>=8.")
print(f"{'Pair':<30} {'p-value':>12} {'Significant (p<0.05)?':>22}")
print("-" * 66)
wilcoxon_results = {}
for a, b in [("NSGA-II","MOEA/D"),
             ("NSGA-II","AGE-MOEA"),
             ("MOEA/D", "AGE-MOEA")]:
    try:
        stat, p = mannwhitneyu(
            hv_stats[a]["values"],
            hv_stats[b]["values"],
            alternative="two-sided"
        )
        sig = "YES" if p < 0.05 else "NO"
    except Exception as e:
        p, sig = 1.0, f"ERROR: {e}"
    label = f"{a} vs {b}"
    wilcoxon_results[label] = {"p": float(p), "significant": sig}
    print(f"{label:<30} {p:>12.6f} {sig:>22}")

# ── Checkpoint: save Section 7.5 results NOW, before the expensive
#    Section 7.6 100-generation convergence runs. If 7.6 crashes or is
#    interrupted, you keep everything computed so far instead of losing
#    it and having to rerun this whole script again.
print("\nCheckpointing Section 7.5 results before starting 7.6...")
saved["hv_stats_30_runs"] = hv_stats
saved["wilcoxon"]         = wilcoxon_results
saved["time_runs"]        = time_runs
saved["n_runs_used"]      = N_RUNS
saved["n_gen_stat"]       = N_GEN_STAT
with open("results_checkpoint_after_7_5.json", "w") as f:
    json.dump(saved, f, indent=2)
print("\u2713 Checkpoint saved to results_checkpoint_after_7_5.json")

# =====================================================================
# SECTION 7.6 — Convergence curves (100 generations)
# =====================================================================
N_GEN_CONV = 100   # ← 100 generations for convergence

print("\n" + "="*60)
print(f"  SECTION 7.6 — CONVERGENCE CURVES ({N_GEN_CONV} generations)")
print("="*60)

class HVCallback(Callback):
    def __init__(self, hv_indicator):
        super().__init__()
        self.hv_ind  = hv_indicator
        self.history = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F").copy()
        F[:, 0] = -F[:, 0]
        try:
            hv_val = self.hv_ind.do(F)
        except Exception:
            hv_val = 0.0
        self.history.append(hv_val)

convergence = {}

# NOTE: all three configs below use final_sampling/final_crossover/
# final_mutation (fix, see note in Step 3) instead of hardcoded types.
algo_configs = [
    ("NSGA-II",
     NSGA2(pop_size=500,
           sampling=final_sampling,
           crossover=final_crossover,
           mutation=final_mutation,
           eliminate_duplicates=True)),
    ("MOEA/D",
     MOEAD(ref_dirs=get_reference_directions("das-dennis", 2, n_partitions=12),
           n_neighbors=15, prob_neighbor_mating=0.9,
           sampling=final_sampling,
           crossover=final_crossover,
           mutation=final_mutation)),
    ("AGE-MOEA",
     AGEMOEA(pop_size=500,
             sampling=final_sampling,
             crossover=final_crossover,
             mutation=final_mutation,
             eliminate_duplicates=True)),
]

for algo_name, algo_obj in algo_configs:
    print(f"  Running {algo_name} ({N_GEN_CONV} gens)...")
    t0 = time.time()
    cb = HVCallback(hv_ind)
    pymoo_minimize(
        problem_final, algo_obj,
        ("n_gen", N_GEN_CONV), seed=42, verbose=False, callback=cb
    )
    elapsed = time.time() - t0
    convergence[algo_name] = cb.history
    print(f"    Final HV: {cb.history[-1]:.6f}  ({elapsed:.0f}s)")

    # Checkpoint after EACH algorithm (not just at the end) -- AGE-MOEA
    # took ~35 min in the original run and is the slowest of the three;
    # if it crashes, this preserves whatever finished before it.
    saved["convergence_100gen"] = convergence
    with open("results_checkpoint_convergence.json", "w") as f:
        json.dump(saved, f, indent=2)
    print(f"    \u2713 Checkpointed after {algo_name}")

print(f"\n--- CONVERGENCE TABLE (every 10 gens) ---")
print(f"{'Gen':<8} {'NSGA-II':>12} {'MOEA/D':>12} {'AGE-MOEA':>12}")
print("-" * 48)
for gen in range(0, N_GEN_CONV, 10):
    print(f"{gen+1:<8} "
          f"{convergence['NSGA-II'][gen]:>12.6f} "
          f"{convergence['MOEA/D'][gen]:>12.6f} "
          f"{convergence['AGE-MOEA'][gen]:>12.6f}")

# ── Save updated results ───────────────────────────────────────────────────
print("\nSaving to results_updated.json...")
saved["hv_stats_30_runs"]   = hv_stats
saved["wilcoxon"]           = wilcoxon_results
saved["convergence_100gen"] = convergence
saved["time_runs"]          = time_runs
saved["n_runs_used"]        = N_RUNS
saved["n_gen_stat"]         = N_GEN_STAT
saved["n_gen_conv"]         = N_GEN_CONV

with open("results_updated.json", "w") as f:
    json.dump(saved, f, indent=2)

print("\u2713 Saved to results_updated.json")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL SUMMARY FOR REPORT")
print("="*60)
print(f"\nStatistical test: {N_RUNS} independent runs, {N_GEN_STAT} generations each")
for algo in ["NSGA-II", "MOEA/D", "AGE-MOEA"]:
    s = hv_stats[algo]
    print(f"  {algo:<12} mean={s['mean']:.6f}  std={s['std']:.6f}  "
          f"min={s['min']:.6f}  max={s['max']:.6f}")

print(f"\nWilcoxon tests ({N_RUNS} runs):")
for pair, res in wilcoxon_results.items():
    print(f"  {pair:<30} p={res['p']:.6f}  {res['significant']}")

print(f"\nConvergence ({N_GEN_CONV} generations, seed=42):")
for algo, hist in convergence.items():
    print(f"  {algo:<12} gen1={hist[0]:.6f}  "
          f"gen50={hist[49]:.6f}  gen100={hist[-1]:.6f}")