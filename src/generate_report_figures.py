"""
Generate report figures from saved pipeline results.

This script reads `results_updated.json`, parses the stored metrics, and
writes all report-ready figures to the `figures/` directory. It does not
run optimisation itself; it is intended to be run after `main_pipeline.py`
or `rerun_test.py` has produced the required JSON keys.

Usage:
    python src/generate_report_figures.py

Expected inputs:
    results_updated.json

Outputs:
    PDF and PNG versions of scaling, Pareto-front, convergence,
    algorithm-comparison, statistical, stress-test, and portfolio
    allocation figures.
"""

import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

os.makedirs("figures", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────
print("Loading results_updated.json...")
with open("results_updated.json") as f:
    R = json.load(f)
print("✓ Loaded\n")

COLORS = {
    "NSGA-II":           "steelblue",
    "MOEA/D":            "darkorange",
    "AGE-MOEA":          "seagreen",
    "Gurobi":            "black",
    "Repair":            "steelblue",
    "Penalty":           "darkorange",
    "Decoder":           "seagreen",
    "SBX + PM":          "steelblue",
    "Uniform Crossover": "darkorange",
    "Single-Point":      "seagreen",
}

# ── Parse ──────────────────────────────────────────────────────────────────
gurobi_returns  = R["gurobi_pareto_front"]["returns"]
gurobi_cvars    = R["gurobi_pareto_front"]["cvars"]
gurobi_F        = np.array([[r, c] for r, c in zip(gurobi_returns, gurobi_cvars)])
gurobi_hv       = R["algorithm_comparison"]["Gurobi"]["HV"]

scaling         = {int(k): v for k, v in R["scaling"].items()}
ch_data         = R["constraint_handling"]
ch_names        = ["Repair", "Penalty", "Decoder"]
ch_curves_gen   = R["constraint_handling_curves"]["generations"]
ch_curves_time  = R["constraint_handling_curves"]["cpu_time"]
op_data         = R["search_operators"]
op_names        = ["SBX + PM", "Uniform Crossover", "Single-Point"]
op_curves_gen   = R["search_operator_curves"]["generations"]
op_curves_time  = R["search_operator_curves"]["cpu_time"]
algo_data       = R["algorithm_comparison"]
conv_data       = R["convergence_100gen"]   # NSGA-II now has real values
hv_stats        = R["hv_stats_30_runs"]
wilcoxon        = R["wilcoxon"]
stress_eq       = R["stress_equal_weight"]
stress_opt      = R["stress_optimised"]
alloc           = R["portfolio_allocation"]
portfolio       = R["best_portfolio"]
time_runs       = R["time_runs"]
n_runs          = R["n_runs_used"]

# Build algorithm convergence time axis from time_runs mean per run
algo_conv_time = {}
for algo in ["NSGA-II", "MOEA/D", "AGE-MOEA"]:
    hvs     = conv_data[algo]
    n       = len(hvs)
    mean_t  = float(np.mean(time_runs[algo]))
    algo_conv_time[algo] = [mean_t * (i + 1) / n for i in range(n)]

# ── Helper: dual plot ──────────────────────────────────────────────────────
def dual_plot(curves_time, curves_gen, gurobi_hv, title, filename,
              xlabel_time="Wall-clock time (s)",
              xlabel_iter="Iteration (generation)"):
    """1×2 figure: left = HV vs CPU time, right = HV vs iteration."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, hvs in curves_gen.items():
        color = COLORS.get(label, "gray")
        times = curves_time[label]
        iters = list(range(1, len(hvs) + 1))
        axes[0].plot(times, hvs, color=color, linewidth=2,
                     label=label, alpha=0.9)
        axes[1].plot(iters, hvs, color=color, linewidth=2,
                     label=label, alpha=0.9)

    for ax, xlabel in zip(axes, [xlabel_time, xlabel_iter]):
        ax.axhline(gurobi_hv, color="black", linestyle="--",
                   linewidth=1.5, label=f"Gurobi (HV={gurobi_hv:.4f})")
        ax.set_ylabel("Hypervolume (HV)", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Solution Quality vs CPU Time", fontsize=11)
    axes[1].set_title("Solution Quality vs Iteration", fontsize=11)
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"figures/{filename}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved {filename}")


# ── FIGURE 1: Gurobi Scaling ───────────────────────────────────────────────
def plot_scaling():
    M_vals = sorted(scaling.keys())
    times  = [scaling[m] for m in M_vals]
    base   = times[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(M_vals, times, "o-", color="steelblue",
            linewidth=2.5, markersize=9)
    ax.fill_between(M_vals, times, alpha=0.08, color="steelblue")

    for x, y in zip(M_vals, times):
        ax.annotate(f"{y:.1f}s", (x, y),
                    textcoords="offset points", xytext=(6, 7), fontsize=9)

    ax.set_xlabel("Universe Size $M$", fontsize=12)
    ax.set_ylabel("Total Sweep Time (seconds)", fontsize=12)
    ax.set_title(
        f"Gurobi $\\varepsilon$-Constraint Scaling\n"
        f"(M=20: {base:.1f}s → M=100: {times[-1]/base:.1f}× slower)",
        fontsize=11
    )
    ax.set_xticks(M_vals)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/gurobi_scaling.pdf", dpi=150)
    plt.savefig("figures/gurobi_scaling.png", dpi=150)
    plt.show()
    print("✓ Saved gurobi_scaling")


# ── FIGURE 2: Gurobi Pareto Front ─────────────────────────────────────────
def plot_pareto_front():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(gurobi_F[:, 1], gurobi_F[:, 0],
            "k*-", markersize=10, linewidth=1.5,
            label=f"Gurobi Exact Front ({len(gurobi_F)} points)")

    # mark best portfolio on the front
    best_ret  = portfolio["expected_return"]
    best_cvar = portfolio["cvar"]
    ax.scatter(best_cvar, best_ret,
               c="red", s=200, zorder=6,
               marker="D", edgecolors="black",
               label=f"Best Portfolio\n(ret={best_ret*100:.1f}%, "
                     f"CVaR={best_cvar*100:.2f}%)")

    ax.set_xlabel("CVaR$_{0.95}$ (Risk)", fontsize=12)
    ax.set_ylabel("Annualised Expected Return", fontsize=12)
    ax.set_title("Gurobi Exact CVaR–Return Pareto Front\n"
                 "(20-point $\\varepsilon$-constraint sweep)", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/pareto_front.pdf", dpi=150)
    plt.savefig("figures/pareto_front.png", dpi=150)
    plt.show()
    print("✓ Saved pareto_front")


# ── FIGURE 3: Constraint Handling Convergence ─────────────────────────────
def plot_constraint_handling_convergence():
    n_gen = len(ch_curves_gen["Repair"])
    dual_plot(
        curves_time = ch_curves_time,
        curves_gen  = ch_curves_gen,
        gurobi_hv   = gurobi_hv,
        title       = "Constraint Handling: Solution Quality vs CPU Time & Iteration\n"
                      f"(NSGA-II, pop=500, {n_gen} generations, seed=42)",
        filename    = "constraint_handling_convergence"
    )


# ── FIGURE 4: Search Operator Convergence ─────────────────────────────────
def plot_operator_convergence():
    n_gen = len(op_curves_gen["SBX + PM"])
    dual_plot(
        curves_time = op_curves_time,
        curves_gen  = op_curves_gen,
        gurobi_hv   = gurobi_hv,
        title       = "Search Operators: Solution Quality vs CPU Time & Iteration\n"
                      f"(NSGA-II + Repair, pop=500, {n_gen} generations, seed=42)",
        filename    = "search_operator_convergence"
    )


# ── FIGURE 5: Algorithm Convergence ───────────────────────────────────────
def plot_algorithm_convergence():
    n_gen = len(conv_data["NSGA-II"])
    dual_plot(
        curves_time = algo_conv_time,
        curves_gen  = {a: conv_data[a]
                       for a in ["NSGA-II", "MOEA/D", "AGE-MOEA"]},
        gurobi_hv   = gurobi_hv,
        title       = "Algorithm Comparison: Solution Quality vs CPU Time & Iteration\n"
                      f"(pop=500, {n_gen} generations, seed=42)\n"
                      "Note: time axis interpolated from mean of 10 independent runs",
        filename    = "algorithm_convergence"
    )


# ── FIGURE 6: Config Summary Bar Charts ───────────────────────────────────
def plot_config_summary():
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)
    bc  = ["steelblue", "darkorange", "seagreen"]

    def _bar(ax, labels, values, colors, title, ylabel, fmt=".4f"):
        bars = ax.bar(labels, values, color=colors, alpha=0.85)
        vmax = max(values) if max(values) > 0 else 1
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + vmax*0.03,
                    f"{v:{fmt}}", ha="center", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    # Row 1: constraint handling
    _bar(fig.add_subplot(gs[0, 0]), ch_names,
         [ch_data[m]["HV"]   for m in ch_names],
         bc, "Constraint Handling\nHV ↑ (higher=better)", "HV")
    _bar(fig.add_subplot(gs[0, 1]), ch_names,
         [ch_data[m]["GD"]   for m in ch_names],
         bc, "Constraint Handling\nGD ↓ (lower=better)", "GD")
    _bar(fig.add_subplot(gs[0, 2]), ch_names,
         [ch_data[m]["time"] for m in ch_names],
         bc, "Constraint Handling\nCPU Time ↓", "Seconds", fmt=".0f")

    # Row 2: search operators
    short = ["SBX+PM", "Uniform", "Single-Pt"]
    _bar(fig.add_subplot(gs[1, 0]), short,
         [op_data[o]["HV"]   for o in op_names],
         bc, "Search Operator\nHV ↑ (higher=better)", "HV")
    _bar(fig.add_subplot(gs[1, 1]), short,
         [op_data[o]["GD"]   for o in op_names],
         bc, "Search Operator\nGD ↓ (lower=better)", "GD")
    _bar(fig.add_subplot(gs[1, 2]), short,
         [op_data[o]["time"] for o in op_names],
         bc, "Search Operator\nCPU Time ↓", "Seconds", fmt=".0f")

    plt.suptitle("Configuration Engineering: Constraint Handling & Search Operator",
                 fontsize=13)
    plt.savefig("figures/config_summary.pdf", dpi=150, bbox_inches="tight")
    plt.savefig("figures/config_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved config_summary")


# ── FIGURE 7: Algorithm Summary ───────────────────────────────────────────
def plot_algorithm_summary():
    algos = ["NSGA-II", "MOEA/D", "AGE-MOEA", "Gurobi"]
    hvs   = [algo_data[a]["HV"]   for a in algos]
    times = [algo_data[a]["time"] for a in algos]
    bc    = [COLORS[a] for a in algos]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: HV bar
    bars = axes[0].bar(algos, hvs, color=bc, alpha=0.85)
    axes[0].axhline(gurobi_hv, color="black", linestyle="--",
                    linewidth=1.2, alpha=0.5, label="Gurobi HV")
    for bar, v in zip(bars, hvs):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     v + max(hvs)*0.02,
                     f"{v:.4f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Hypervolume (HV) ↑", fontsize=11)
    axes[0].set_title("Final HV by Algorithm", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: HV vs Time scatter (log x)
    label_offsets = {
        "NSGA-II":  (-55,  8),
        "MOEA/D":   ( 10,  8),
        "AGE-MOEA": ( 10, -15),
        "Gurobi":   ( 10,  8),
    }
    for algo, hv, t in zip(algos, hvs, times):
        axes[1].scatter(t, hv, c=COLORS[algo], s=200,
                        zorder=4, edgecolors="black", linewidths=0.5)
        dx, dy = label_offsets[algo]
        axes[1].annotate(f"{algo}\n({t:.0f}s)", (t, hv),
                         xytext=(dx, dy),
                         textcoords="offset points",
                         fontsize=8.5)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Wall-clock Time (s, log scale)", fontsize=11)
    axes[1].set_ylabel("Hypervolume (HV)", fontsize=11)
    axes[1].set_title("HV vs CPU Time Trade-off\n"
                      "(upper-left = best: high HV, low time)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Final Algorithm Comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/algorithm_summary.pdf", dpi=150)
    plt.savefig("figures/algorithm_summary.png", dpi=150)
    plt.show()
    print("✓ Saved algorithm_summary")


# ── FIGURE 8: Statistical Boxplot + Wilcoxon ──────────────────────────────
def plot_statistical_boxplot():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    algos      = ["NSGA-II", "MOEA/D", "AGE-MOEA"]
    data       = [hv_stats[a]["values"] for a in algos]
    bp         = axes[0].boxplot(data, patch_artist=True,
                                  medianprops={"color": "black",
                                               "linewidth": 2})
    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.7)

    axes[0].axhline(gurobi_hv, color="black", linestyle="--",
                    linewidth=1.5, label=f"Gurobi HV={gurobi_hv:.4f}")
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xticklabels(algos, fontsize=11)
    axes[0].set_ylabel("Hypervolume (HV)", fontsize=12)
    axes[0].set_title(
        f"HV Distribution ({n_runs} independent runs)\n"
        f"NSGA-II mean={hv_stats['NSGA-II']['mean']:.4f}  "
        f"AGE-MOEA mean={hv_stats['AGE-MOEA']['mean']:.4f}  "
        f"MOEA/D mean={hv_stats['MOEA/D']['mean']:.4f}",
        fontsize=10
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: Wilcoxon p-values
    pairs      = list(wilcoxon.keys())
    pvals      = [wilcoxon[p]["p"] for p in pairs]
    sigs       = [wilcoxon[p]["significant"] for p in pairs]
    colors_p   = ["seagreen" if s == "YES" else "salmon" for s in sigs]
    pair_short = ["NSGA-II\nvs MOEA/D",
                  "NSGA-II\nvs AGE-MOEA",
                  "MOEA/D\nvs AGE-MOEA"]

    bars = axes[1].bar(pair_short, pvals, color=colors_p, alpha=0.85)
    axes[1].axhline(0.05, color="red", linestyle="--",
                    linewidth=1.5, label="$\\alpha=0.05$ threshold")

    for bar, v, s in zip(bars, pvals, sigs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     max(v + max(pvals)*0.02, 0.001),
                     f"p={v:.4f}\n({'sig.' if s=='YES' else 'not sig.'})",
                     ha="center", fontsize=9)

    axes[1].set_ylabel("p-value (Mann-Whitney U)", fontsize=12)
    axes[1].set_title("Wilcoxon Rank-Sum Test p-values\n"
                      "(green = statistically significant at p<0.05)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Statistical Significance: {n_runs} Independent Runs",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/statistical_boxplot.pdf", dpi=150)
    plt.savefig("figures/statistical_boxplot.png", dpi=150)
    plt.show()
    print("✓ Saved statistical_boxplot")


# ── FIGURE 9: Stress Test ─────────────────────────────────────────────────
def plot_stress_test():
    covid_eq   = stress_eq["covid"]
    tariff_eq  = stress_eq["tariff"]
    covid_opt  = stress_opt["covid"]
    tariff_opt = stress_opt["tariff"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    windows   = ["COVID-19\n(Feb–Apr 2020)", "2025 Tariff\nShock"]
    metrics   = [
        ("Realised CVaR",  "realised_cvar", False),
        ("Max Drawdown",   "max_drawdown",  False),
        ("Sharpe Ratio",   "sharpe_ratio",  True),
    ]

    for ax, (label, key, higher_better) in zip(axes, metrics):
        eq_vals  = [abs(covid_eq[key]),  abs(tariff_eq[key])]
        opt_vals = [abs(covid_opt[key]), abs(tariff_opt[key])]
        x        = np.arange(len(windows))
        w        = 0.35

        bars1 = ax.bar(x - w/2, eq_vals,  w, label="Equal Weight (1/N)",
                       color="salmon",    alpha=0.85)
        bars2 = ax.bar(x + w/2, opt_vals, w, label="Optimised Portfolio",
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
        ax.set_ylabel("↓ lower is better" if not higher_better
                      else "↑ higher is better", fontsize=9)

    plt.suptitle("Out-of-Sample Stress Test: Equal Weight vs Optimised Portfolio",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/stress_test.pdf", dpi=150)
    plt.savefig("figures/stress_test.png", dpi=150)
    plt.show()
    print("✓ Saved stress_test")


# ── FIGURE 10: Portfolio Pie Chart ────────────────────────────────────────
def plot_portfolio_weights():
    color_map = {
        "Bond ETF":       "#4878CF",
        "Cryptocurrency": "#D65F5F",
        "Broad ETF":      "#6ACC65",
        "S&P 500 Equity": "#B47CC7",
    }
    labels     = []
    weights    = []
    colors_pie = []

    for item in alloc:
        labels.append(f"{item['ticker']}\n({item['weight']*100:.1f}%)")
        weights.append(item["weight"])
        colors_pie.append(color_map.get(item["asset_class"], "#999999"))

    fig, ax = plt.subplots(figsize=(9, 7))
    _, texts, autotexts = ax.pie(
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
        f"Optimal Portfolio Allocation — {portfolio['n_assets']} Assets\n"
        f"Return={portfolio['expected_return']*100:.2f}%  "
        f"CVaR={portfolio['cvar']*100:.2f}%  "
        f"Sharpe={portfolio['realised_sharpe']:.2f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("figures/portfolio_weights.pdf", dpi=150)
    plt.savefig("figures/portfolio_weights.png", dpi=150)
    plt.show()
    print("✓ Saved portfolio_weights")


# ── Run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  GENERATING ALL FIGURES FROM results.json")
    print("=" * 55 + "\n")

    plot_scaling()
    plot_pareto_front()
    plot_constraint_handling_convergence()
    plot_operator_convergence()
    plot_algorithm_convergence()
    plot_config_summary()
    plot_algorithm_summary()
    plot_statistical_boxplot()
    plot_stress_test()
    plot_portfolio_weights()

    print("\n" + "=" * 55)
    print("  DONE — all figures saved to figures/")
    print("=" * 55)
    print("""
  gurobi_scaling.pdf
  pareto_front.pdf
  constraint_handling_convergence.pdf
  search_operator_convergence.pdf
  algorithm_convergence.pdf
  config_summary.pdf
  algorithm_summary.pdf
  statistical_boxplot.pdf
  stress_test.pdf
  portfolio_weights.pdf
    """)
