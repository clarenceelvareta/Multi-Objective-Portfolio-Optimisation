# plot_results.py
# Run this in your notebook AFTER main_pipeline.py finishes
# All variables (gurobi_F, nsga_F, moead_F, age_F, etc.) must exist in memory

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import time
import os

os.makedirs("figures", exist_ok=True)

COLORS = {
    "NSGA-II":  "steelblue",
    "MOEA/D":   "darkorange",
    "AGE-MOEA": "seagreen",
    "Gurobi":   "black",
}

# ── FIGURE 1: Gurobi Scaling ───────────────────────────────────────────────
def plot_scaling(scaling_results):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    M_vals = list(scaling_results.keys())
    times  = list(scaling_results.values())

    ax.plot(M_vals, times, "o-", color="steelblue",
            linewidth=2.5, markersize=9, zorder=3)
    ax.fill_between(M_vals, times, alpha=0.08, color="steelblue")

    for x, y in zip(M_vals, times):
        ax.annotate(f"{y:.1f}s", (x, y),
                    textcoords="offset points",
                    xytext=(6, 7), fontsize=9)

    ax.set_xlabel("Universe Size $M$", fontsize=12)
    ax.set_ylabel("Total Sweep Time (seconds)", fontsize=12)
    ax.set_title("Gurobi $\\varepsilon$-Constraint Scaling\n"
                 "(20 epsilon points per $M$, strict $K=15$ cardinality)",
                 fontsize=11)
    ax.set_xticks(M_vals)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/gurobi_scaling.pdf", dpi=150)
    plt.savefig("figures/gurobi_scaling.png", dpi=150)
    plt.show()
    print("✓ Saved gurobi_scaling")


# ── FIGURE 2: Pareto Front Comparison ─────────────────────────────────────
def plot_pareto_front(gurobi_F, nsga_F, moead_F, age_F):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(moead_F[:, 1],  moead_F[:, 0],
               c=COLORS["MOEA/D"],   marker="s", s=35,
               alpha=0.6, label="MOEA/D",   zorder=2)
    ax.scatter(age_F[:, 1],    age_F[:, 0],
               c=COLORS["AGE-MOEA"], marker="^", s=35,
               alpha=0.6, label="AGE-MOEA", zorder=3)
    ax.scatter(nsga_F[:, 1],   nsga_F[:, 0],
               c=COLORS["NSGA-II"],  marker="o", s=35,
               alpha=0.6, label="NSGA-II",  zorder=4)
    ax.scatter(gurobi_F[:, 1], gurobi_F[:, 0],
               c=COLORS["Gurobi"],   marker="*", s=180,
               zorder=5, label="Gurobi (Exact)")

    ax.set_xlabel("CVaR$_{0.95}$ (Risk)", fontsize=12)
    ax.set_ylabel("Annualised Expected Return", fontsize=12)
    ax.set_title("CVaR–Return Pareto Front: Algorithm Comparison", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/pareto_front.pdf", dpi=150)
    plt.savefig("figures/pareto_front.png", dpi=150)
    plt.show()
    print("✓ Saved pareto_front")


# ── FIGURE 3: Solution Value vs CPU Time ──────────────────────────────────
# Uses the 30-run time_runs and hv_runs already computed in Section 7.5
def plot_solution_value_vs_cpu_time(problem_final, hv_ind, gurobi_F):
    """
    Tracks HV continuously over wall-clock time during a single long run.
    Produces a smooth curve like the reference image.
    """
    from pymoo.algorithms.moo.nsga2    import NSGA2
    from pymoo.algorithms.moo.moead    import MOEAD
    from pymoo.algorithms.moo.age      import AGEMOEA
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm   import PM
    from pymoo.operators.sampling.rnd  import IntegerRandomSampling, FloatRandomSampling
    from pymoo.optimize                import minimize as pymoo_minimize
    from pymoo.util.ref_dirs           import get_reference_directions
    from pymoo.core.callback           import Callback

    POP_SIZE = 500
    N_GEN    = 200  # run long enough for curve to flatten

    class TimeHVCallback(Callback):
        """Records (wall_clock_time, HV) at every generation."""
        def __init__(self, hv_indicator, start_time):
            super().__init__()
            self.hv_ind     = hv_indicator
            self.start_time = start_time
            self.time_hist  = []
            self.hv_hist    = []

        def notify(self, algorithm):
            elapsed = time.time() - self.start_time
            F = algorithm.pop.get("F").copy()
            F[:, 0] = -F[:, 0]
            try:
                hv_val = self.hv_ind.do(F)
            except Exception:
                hv_val = 0.0
            self.time_hist.append(elapsed)
            self.hv_hist.append(hv_val)

    convergence_time = {}

    runs = [
        ("NSGA-II",
         NSGA2(pop_size=POP_SIZE,
               sampling=FloatRandomSampling(),
               crossover=SBX(prob=0.9, eta=15),
               mutation=PM(eta=20),
               eliminate_duplicates=True)),
        ("MOEA/D",
         MOEAD(ref_dirs=get_reference_directions(
                   "das-dennis", 2, n_partitions=12),
               n_neighbors=15, prob_neighbor_mating=0.9,
               sampling=IntegerRandomSampling(),
               crossover=SBX(prob=0.9, eta=15, vtype=int),
               mutation=PM(eta=20, vtype=int))),
        ("AGE-MOEA",
         AGEMOEA(pop_size=POP_SIZE,
                 sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15, vtype=int),
                 mutation=PM(eta=20, vtype=int),
                 eliminate_duplicates=True)),
    ]

    for algo_name, algo_obj in runs:
        print(f"  Running {algo_name} ({N_GEN} gens)...")
        t0 = time.time()
        cb = TimeHVCallback(hv_ind, start_time=t0)
        pymoo_minimize(
            problem_final, algo_obj,
            ("n_gen", N_GEN), seed=42, verbose=False, callback=cb
        )
        convergence_time[algo_name] = {
            "times": cb.time_hist,
            "hvs":   cb.hv_hist,
        }
        print(f"    Final HV={cb.hv_hist[-1]:.4f}  "
              f"Total time={cb.time_hist[-1]:.1f}s")

    gurobi_hv = hv_ind.do(gurobi_F)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    for algo, data in convergence_time.items():
        ax.plot(data["times"], data["hvs"],
                color=COLORS[algo], linewidth=2,
                label=algo, alpha=0.9)

    ax.axhline(gurobi_hv, color=COLORS["Gurobi"],
               linestyle="--", linewidth=1.8,
               label=f"Gurobi (HV={gurobi_hv:.4f})")

    ax.set_xlabel("CPU Time (seconds)", fontsize=12)
    ax.set_ylabel("Hypervolume (HV)", fontsize=12)
    ax.set_title("Solution Quality vs CPU Time\n"
                 f"(population={POP_SIZE}, {N_GEN} generations, seed=42)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/solution_value_vs_cpu_time.pdf", dpi=150)
    plt.savefig("figures/solution_value_vs_cpu_time.png", dpi=150)
    plt.show()
    print("✓ Saved solution_value_vs_cpu_time")

    return convergence_time


# ── FIGURE 4: Solution Value vs Iteration (>50,000 evaluations) ───────────
def plot_solution_value_vs_iteration(problem_final, hv_ind, gurobi_F):
    from pymoo.algorithms.moo.nsga2    import NSGA2
    from pymoo.algorithms.moo.moead    import MOEAD
    from pymoo.algorithms.moo.age      import AGEMOEA
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm   import PM
    from pymoo.operators.sampling.rnd  import IntegerRandomSampling, FloatRandomSampling
    from pymoo.optimize                import minimize as pymoo_minimize
    from pymoo.util.ref_dirs           import get_reference_directions
    from pymoo.core.callback           import Callback

    POP_SIZE = 500
    N_GEN    = 200   # 500 × 200 = 100,000 evaluations

    class HVEvalCallback(Callback):
        def __init__(self, hv_indicator):
            super().__init__()
            self.hv_ind    = hv_indicator
            self.hv_hist   = []
            self.eval_hist = []
            self.n_eval    = 0

        def notify(self, algorithm):
            self.n_eval += POP_SIZE
            F = algorithm.pop.get("F").copy()
            F[:, 0] = -F[:, 0]
            try:
                hv_val = self.hv_ind.do(F)
            except Exception:
                hv_val = 0.0
            self.hv_hist.append(hv_val)
            self.eval_hist.append(self.n_eval)

    convergence_eval = {}

    runs = [
        ("NSGA-II",
         NSGA2(pop_size=POP_SIZE,
               sampling=FloatRandomSampling(),
               crossover=SBX(prob=0.9, eta=15),
               mutation=PM(eta=20),
               eliminate_duplicates=True)),
        ("MOEA/D",
         MOEAD(ref_dirs=get_reference_directions("das-dennis", 2, n_partitions=12),
               n_neighbors=15, prob_neighbor_mating=0.9,
               sampling=IntegerRandomSampling(),
               crossover=SBX(prob=0.9, eta=15, vtype=int),
               mutation=PM(eta=20, vtype=int))),
        ("AGE-MOEA",
         AGEMOEA(pop_size=POP_SIZE,
                 sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15, vtype=int),
                 mutation=PM(eta=20, vtype=int),
                 eliminate_duplicates=True)),
    ]

    for algo_name, algo_obj in runs:
        print(f"  Running {algo_name} for {N_GEN} gens ({N_GEN*POP_SIZE:,} evals)...")
        cb = HVEvalCallback(hv_ind)
        pymoo_minimize(
            problem_final, algo_obj,
            ("n_gen", N_GEN), seed=42, verbose=False, callback=cb
        )
        convergence_eval[algo_name] = {
            "evals": cb.eval_hist,
            "hvs":   cb.hv_hist
        }
        print(f"    Final HV: {cb.hv_hist[-1]:.6f}")

    gurobi_hv = hv_ind.do(gurobi_F)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full 100,000 evaluations
    for algo, data in convergence_eval.items():
        axes[0].plot(data["evals"], data["hvs"],
                     color=COLORS[algo], linewidth=2,
                     label=algo, alpha=0.85)
    axes[0].axhline(gurobi_hv, color=COLORS["Gurobi"],
                    linestyle="--", linewidth=1.5,
                    label=f"Gurobi (HV={gurobi_hv:.4f})")
    axes[0].axvline(50000, color="gray", linestyle=":",
                    linewidth=1.2, label="50,000 evaluations")
    axes[0].set_xlabel("Function Evaluations", fontsize=12)
    axes[0].set_ylabel("Hypervolume (HV)", fontsize=12)
    axes[0].set_title(f"Solution Value vs Evaluations\n"
                      f"(Full run: {N_GEN*POP_SIZE:,} evaluations)",
                      fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    # Right: zoom into first 50,000
    for algo, data in convergence_eval.items():
        evals = np.array(data["evals"])
        hvs   = np.array(data["hvs"])
        mask  = evals <= 50000
        axes[1].plot(evals[mask], hvs[mask],
                     color=COLORS[algo], linewidth=2,
                     label=algo, alpha=0.85)
    axes[1].axhline(gurobi_hv, color=COLORS["Gurobi"],
                    linestyle="--", linewidth=1.5,
                    label=f"Gurobi (HV={gurobi_hv:.4f})")
    axes[1].set_xlabel("Function Evaluations", fontsize=12)
    axes[1].set_ylabel("Hypervolume (HV)", fontsize=12)
    axes[1].set_title("Solution Value vs Evaluations\n"
                      "(Zoom: first 50,000 evaluations)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.suptitle("Convergence: Hypervolume vs Number of Function Evaluations",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("figures/solution_value_vs_iteration.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig("figures/solution_value_vs_iteration.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved solution_value_vs_iteration")
    return convergence_eval

print("\nGenerating solution value vs iteration plot (100,000 evaluations)...")


# ── FIGURE 5: HV Convergence per Generation ───────────────────────────────
def plot_convergence_hv(convergence, N_GEN, gurobi_F, hv_ind):
    if convergence["NSGA-II"][-1] == 0.0:
        print("Skipping convergence_hv — Section 7.6 not run yet")
        return

    gurobi_hv = hv_ind.do(gurobi_F)
    gens      = range(1, N_GEN + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, hist in convergence.items():
        ax.plot(gens, hist, color=COLORS[algo],
                linewidth=2, label=algo, alpha=0.85)
    ax.axhline(gurobi_hv, color=COLORS["Gurobi"],
               linestyle="--", linewidth=1.5,
               label=f"Gurobi reference (HV={gurobi_hv:.4f})")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Hypervolume (HV)", fontsize=12)
    ax.set_title("Hypervolume Convergence over Generations\n"
                 f"(population=500, {N_GEN} generations)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/convergence_hv.pdf", dpi=150)
    plt.savefig("figures/convergence_hv.png", dpi=150)
    plt.show()
    print("✓ Saved convergence_hv")



# ── FIGURE 6: Constraint Handling & Operator Comparison ───────────────────
def plot_config_comparison(constraint_results, operator_results):
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    methods    = list(constraint_results.keys())
    ops        = list(operator_results.keys())
    op_short   = ["SBX+PM", "Uniform", "Single-Pt"]
    bar_colors = ["steelblue", "darkorange", "seagreen"]

    # Row 1: Constraint handling — HV, GD, Time
    ch_hvs   = [constraint_results[m]["HV"]   for m in methods]
    ch_gds   = [constraint_results[m]["GD"]   for m in methods]
    ch_times = [constraint_results[m]["time"] for m in methods]

    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(methods, ch_hvs, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, ch_hvs):
        ax0.text(bar.get_x() + bar.get_width()/2,
                 v + max(ch_hvs)*0.02,
                 f"{v:.4f}", ha="center", fontsize=9)
    ax0.set_title("Constraint Handling\nHV (higher=better)", fontsize=10)
    ax0.set_ylabel("HV", fontsize=10)
    ax0.grid(True, alpha=0.3, axis="y")

    ax1 = fig.add_subplot(gs[0, 1])
    bars = ax1.bar(methods, ch_gds, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, ch_gds):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 v + max(ch_gds)*0.02,
                 f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_title("Constraint Handling\nGD (lower=better)", fontsize=10)
    ax1.set_ylabel("GD", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(methods, ch_times, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, ch_times):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 v + max(ch_times)*0.02,
                 f"{v:.0f}s", ha="center", fontsize=9)
    ax2.set_title("Constraint Handling\nCPU Time (lower=better)", fontsize=10)
    ax2.set_ylabel("Seconds", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # Row 2: Search operators — HV, GD, Time
    op_hvs   = [operator_results[o]["HV"]   for o in ops]
    op_gds   = [operator_results[o]["GD"]   for o in ops]
    op_times = [operator_results[o]["time"] for o in ops]

    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(op_short, op_hvs, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, op_hvs):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 v + max(op_hvs)*0.02,
                 f"{v:.4f}", ha="center", fontsize=9)
    ax3.set_title("Search Operator\nHV (higher=better)", fontsize=10)
    ax3.set_ylabel("HV", fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(op_short, op_gds, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, op_gds):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 v + max(op_gds)*0.02,
                 f"{v:.3f}", ha="center", fontsize=9)
    ax4.set_title("Search Operator\nGD (lower=better)", fontsize=10)
    ax4.set_ylabel("GD", fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")

    ax5 = fig.add_subplot(gs[1, 2])
    bars = ax5.bar(op_short, op_times, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, op_times):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 v + max(op_times)*0.02,
                 f"{v:.0f}s", ha="center", fontsize=9)
    ax5.set_title("Search Operator\nCPU Time (lower=better)", fontsize=10)
    ax5.set_ylabel("Seconds", fontsize=10)
    ax5.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Configuration Engineering: Constraint Handling vs Search Operator",
                 fontsize=13, y=1.01)
    plt.savefig("figures/config_comparison.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig("figures/config_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved config_comparison")



# ── FIGURE 7: Statistical Significance — Boxplot (30 runs) ───────────────
def plot_statistical_boxplot(hv_runs, hv_stats, wilcoxon_results, gurobi_F, hv_ind):
    if hv_stats["NSGA-II"]["mean"] == 0.0:
        print("Skipping boxplot — Section 7.5 not run yet")
        return

    gurobi_hv = hv_ind.do(gurobi_F)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Boxplot
    data   = [hv_runs[a] for a in ["NSGA-II", "MOEA/D", "AGE-MOEA"]]
    bp     = axes[0].boxplot(data, patch_artist=True,
                              medianprops={"color": "black", "linewidth": 2})
    colors_box = [COLORS["NSGA-II"], COLORS["MOEA/D"], COLORS["AGE-MOEA"]]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].axhline(gurobi_hv, color=COLORS["Gurobi"],
                    linestyle="--", linewidth=1.5,
                    label=f"Gurobi HV={gurobi_hv:.4f}")
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xticklabels(["NSGA-II", "MOEA/D", "AGE-MOEA"], fontsize=11)
    axes[0].set_ylabel("Hypervolume (HV)", fontsize=12)
    axes[0].set_title("HV Distribution over 30 Independent Runs", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: Wilcoxon p-values
    pairs  = list(wilcoxon_results.keys())
    pvals  = [wilcoxon_results[p]["p"]   for p in pairs]
    sigs   = [wilcoxon_results[p]["significant"] for p in pairs]
    colors_p = ["seagreen" if s == "YES" else "salmon" for s in sigs]
    pair_short = ["NSGA-II\nvs MOEA/D",
                  "NSGA-II\nvs AGE-MOEA",
                  "MOEA/D\nvs AGE-MOEA"]

    bars = axes[1].bar(pair_short, pvals, color=colors_p, alpha=0.85)
    axes[1].axhline(0.05, color="red", linestyle="--",
                    linewidth=1.5, label="$\\alpha=0.05$ threshold")

    for bar, v, s in zip(bars, pvals, sigs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     v + 0.001,
                     f"p={v:.4f}\n({'sig.' if s=='YES' else 'not sig.'})",
                     ha="center", fontsize=9)

    axes[1].set_ylabel("p-value (Mann-Whitney U)", fontsize=12)
    axes[1].set_title("Wilcoxon Rank-Sum Test p-values\n"
                      "(green = significant difference)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Statistical Significance: 30 Independent Runs",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/statistical_boxplot.pdf", dpi=150)
    plt.savefig("figures/statistical_boxplot.png", dpi=150)
    plt.show()
    print("✓ Saved statistical_boxplot")


# ── FIGURE 8: Stress Test ─────────────────────────────────────────────────
def plot_stress_test(covid_eq, covid_opt, tariff_eq, tariff_opt):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    windows  = ["COVID-19\n(Feb–Apr 2020)", "2025 Tariff\nShock"]
    metrics  = [
        ("Realised CVaR",  "realised_cvar", False),
        ("Max Drawdown",   "max_drawdown",  False),
        ("Sharpe Ratio",   "sharpe_ratio",  True),
    ]

    for ax, (label, key, higher_better) in zip(axes, metrics):
        eq_vals  = [abs(covid_eq[key]),  abs(tariff_eq[key])]
        opt_vals = [abs(covid_opt[key]), abs(tariff_opt[key])]
        x        = np.arange(len(windows))
        w        = 0.35

        bars1 = ax.bar(x - w/2, eq_vals,  w,
                       label="Equal Weight",
                       color="salmon",    alpha=0.85)
        bars2 = ax.bar(x + w/2, opt_vals, w,
                       label="Optimised",
                       color="steelblue", alpha=0.85)

        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(eq_vals)*0.02,
                    f"{bar.get_height():.3f}",
                    ha="center", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(windows, fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylabel("lower is better" if not higher_better
                      else "higher is better", fontsize=9)

    plt.suptitle("Out-of-Sample Stress Test: Equal Weight vs Optimised Portfolio",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/stress_test.pdf", dpi=150)
    plt.savefig("figures/stress_test.png", dpi=150)
    plt.show()
    print("✓ Saved stress_test")



# ── FIGURE 9: Portfolio Weight Pie Chart ──────────────────────────────────
def plot_portfolio_weights(best_portfolio, tickers, w_best,
                            BOND_ETFS, CRYPTOS, BROAD_ETFS):
    labels     = []
    weights    = []
    colors_pie = []

    color_map = {
        "Bond ETF":       "#4878CF",
        "Cryptocurrency": "#D65F5F",
        "Broad ETF":      "#6ACC65",
        "S&P 500 Equity": "#B47CC7",
    }

    for i, t in enumerate(tickers):
        if best_portfolio["z"][i] == 1:
            w_i = w_best[i]
            if t in BOND_ETFS:
                ac = "Bond ETF"
            elif t in CRYPTOS:
                ac = "Cryptocurrency"
            elif t in BROAD_ETFS:
                ac = "Broad ETF"
            else:
                ac = "S&P 500 Equity"
            labels.append(f"{t}\n({w_i*100:.1f}%)")
            weights.append(w_i)
            colors_pie.append(color_map[ac])

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        weights, labels=labels, colors=colors_pie,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75, labeldistance=1.12,
        textprops={"fontsize": 8.5}
    )
    for at in autotexts:
        at.set_fontsize(7.5)

    handles = [mpatches.Patch(color=c, label=l)
               for l, c in color_map.items()]
    ax.legend(handles=handles, loc="lower left",
              fontsize=9, title="Asset Class")
    ax.set_title(
        f"Optimal Portfolio Allocation\n"
        f"(Return={best_portfolio['ret']*100:.2f}%,  "
        f"CVaR={best_portfolio['cvar']*100:.2f}%)",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("figures/portfolio_weights.pdf", dpi=150)
    plt.savefig("figures/portfolio_weights.png", dpi=150)
    plt.show()
    print("✓ Saved portfolio_weights")


# ── SUMMARY ────────────────────────────────────────────────────────────────
# --- Replace everything at the bottom of generate_result_figures.py ---

def generate_all_figures(results):
    # Unpack the dictionary
    gurobi_F = results["gurobi_F"]
    nsga_F = results["nsga_F"]
    moead_F = results["moead_F"]
    age_F = results["age_F"]
    scaling_results = results["scaling_results"]
    hv_runs = results["hv_runs"]
    hv_stats = results["hv_stats"]
    wilcoxon_results = results["wilcoxon_results"]
    hv_ind = results["hv_ind"]
    convergence = results["convergence"]
    N_GEN = results["N_GEN"]
    constraint_results = results["constraint_results"]
    operator_results = results["operator_results"]
    covid_eq = results["covid_eq"]
    covid_opt = results["covid_opt"]
    tariff_eq = results["tariff_eq"]
    tariff_opt = results["tariff_opt"]
    best_portfolio = results["best_portfolio"]
    tickers = results["tickers"]
    w_best = results["w_best"]
    BOND_ETFS = results["BOND_ETFS"]
    CRYPTOS = results["CRYPTOS"]
    BROAD_ETFS = results["BROAD_ETFS"]
    problem_final = results["problem_final"]
    algo_metrics = results["algo_metrics"]

    # Now call your plots
    plot_scaling(scaling_results)
    plot_pareto_front(gurobi_F, nsga_F, moead_F, age_F)
    plot_solution_value_vs_cpu_time(problem_final, hv_ind, gurobi_F) # Note: check your function signature for this one!
    plot_solution_value_vs_iteration(problem_final, hv_ind, gurobi_F)
    plot_convergence_hv(convergence, N_GEN, gurobi_F, hv_ind)
    plot_config_comparison(constraint_results, operator_results)
    plot_statistical_boxplot(hv_runs, hv_stats, wilcoxon_results, gurobi_F, hv_ind)
    plot_stress_test(covid_eq, covid_opt, tariff_eq, tariff_opt)
    plot_portfolio_weights(best_portfolio, tickers, w_best, BOND_ETFS, CRYPTOS, BROAD_ETFS)

    print("\n[All figures saved to figures/]")
    print("\n" + "=" * 55)
    print("ALL FIGURES SAVED TO figures/")
    print("=" * 55)

if __name__ == "__main__":
    # If you run this script directly from terminal, it will import and run main_pipeline first
    from main_pipeline import main
    print("Running main_pipeline.py first...")
    pipeline_results = main()
    print("Generating figures...")
    generate_all_figures(pipeline_results)