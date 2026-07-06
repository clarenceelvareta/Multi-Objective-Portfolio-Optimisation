# src/run_gurobi_scaling_frontier.py

from pathlib import Path
import json
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from config import (
    SP500_EQUITIES,
    BOND_ETFS,
    BROAD_ETFS,
    CRYPTOS,
    SECTOR_MAP,
    K,
    W_MIN,
    W_MAX,
    SECTOR_CAP,
    CRYPTO_CAP,
    BOND_FLOOR,
    CVAR_ALPHA,
)


# ============================================================
# USER SETTINGS
# ============================================================

ALPHA = CVAR_ALPHA

# Scaling sizes
# NOTE: 150 and 200 will be skipped unless prices.csv has enough assets.
M_LIST = [20, 40, 60, 80, 100, 150, 200]

# Common full 20-point epsilon frontier
# This keeps the same risk thresholds across all M values.
EPS_GRID = np.linspace(0.010, 0.045, 20)

# Gurobi controls
TIME_LIMIT_PER_EPS = 300
MIP_GAP_TARGET = 1e-4

# Paths
ROOT = Path(__file__).resolve().parents[1]
PRICE_FILE = ROOT / "src" / "raw" / "prices.csv"
OUT_DIR = ROOT / "scaling_results"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_prices() -> pd.DataFrame:
    """
    Load cached adjusted-close prices from src/raw/prices.csv.
    """
    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {PRICE_FILE}. "
            "Make sure src/raw/prices.csv exists before running this script."
        )

    prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    prices.columns = [str(c).strip() for c in prices.columns]

    # Convert safely to numeric
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Drop assets with no data at all
    prices = prices.dropna(axis=1, how="all")

    return prices


def get_asset_order(prices: pd.DataFrame) -> list:
    """
    Preserve the intended project universe order:
    equities -> bonds -> broad ETFs -> crypto,
    then append any leftover columns from prices.csv.
    """
    available = list(prices.columns)

    intended_order = SP500_EQUITIES + BOND_ETFS + BROAD_ETFS + CRYPTOS
    ordered = [t for t in intended_order if t in available]
    leftovers = [t for t in available if t not in ordered]

    return ordered + leftovers


def compute_returns(prices_subset: pd.DataFrame):
    """
    Compute log returns, annualised mean returns, and daily scenario matrix.
    """
    clean_prices = prices_subset.ffill().dropna(axis=0, how="any")
    returns = np.log(clean_prices / clean_prices.shift(1)).dropna(axis=0, how="any")

    if returns.empty:
        raise ValueError("Return matrix is empty after alignment.")

    mu_annual = 252 * returns.mean().values
    R_daily = returns.values

    return returns, mu_annual, R_daily


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

def select_assets_for_M(all_assets: list, M: int):
    """
    Stratified asset selection for the Gurobi scaling experiment.

    Instead of taking the first M tickers, this preserves the approximate
    full-universe mix:

        80 equities, 6 bonds, 10 broad ETFs, 4 cryptos.

    This makes M=20,40,60,80,100 comparable because each sub-universe
    still contains:
        - equity sector caps,
        - bond floor,
        - crypto cap,
        - broad ETF exposure.
    """
    if M > len(all_assets):
        return None

    available = set(all_assets)

    equities = [t for t in SP500_EQUITIES if t in available]
    bonds = [t for t in BOND_ETFS if t in available]
    etfs = [t for t in BROAD_ETFS if t in available]
    cryptos = [t for t in CRYPTOS if t in available]

    groups = {
        "equity": equities,
        "bond": bonds,
        "etf": etfs,
        "crypto": cryptos,
    }

    full_total = (
        len(SP500_EQUITIES)
        + len(BOND_ETFS)
        + len(BROAD_ETFS)
        + len(CRYPTOS)
    )

    target_fracs = {
        "equity": len(SP500_EQUITIES) / full_total,
        "bond": len(BOND_ETFS) / full_total,
        "etf": len(BROAD_ETFS) / full_total,
        "crypto": len(CRYPTOS) / full_total,
    }

    counts = {
        name: int(round(M * frac))
        for name, frac in target_fracs.items()
    }

    # Ensure important constraint-bearing classes appear when possible.
    if len(equities) > 0:
        counts["equity"] = max(1, counts["equity"])

    if BOND_FLOOR > 0 and len(bonds) > 0:
        counts["bond"] = max(1, counts["bond"])

    if len(etfs) > 0:
        counts["etf"] = max(1, counts["etf"])

    if CRYPTO_CAP > 0 and len(cryptos) > 0:
        counts["crypto"] = max(1, counts["crypto"])

    # Do not request more from a class than available.
    for name in counts:
        counts[name] = min(counts[name], len(groups[name]))

    def total_count():
        return sum(counts.values())

    # If rounding overshoots, remove mainly from equities first.
    remove_order = ["equity", "etf", "crypto", "bond"]
    while total_count() > M:
        removed = False
        for name in remove_order:
            min_allowed = 1 if len(groups[name]) > 0 else 0
            if counts[name] > min_allowed:
                counts[name] -= 1
                removed = True
                break
        if not removed:
            break

    # If rounding undershoots, add mainly to equities first.
    add_order = ["equity", "etf", "bond", "crypto"]
    while total_count() < M:
        added = False
        for name in add_order:
            if counts[name] < len(groups[name]):
                counts[name] += 1
                added = True
                break
        if not added:
            break

    selected = (
        equities[:counts["equity"]]
        + bonds[:counts["bond"]]
        + etfs[:counts["etf"]]
        + cryptos[:counts["crypto"]]
    )

    # Remove duplicates while preserving order.
    selected = list(dict.fromkeys(selected))

    # Final top-up from any available asset if needed.
    if len(selected) < M:
        for t in all_assets:
            if t not in selected:
                selected.append(t)
            if len(selected) == M:
                break

    if len(selected) != M:
        raise ValueError(
            f"Stratified sampling failed for M={M}. "
            f"Selected {len(selected)} assets instead. Counts={counts}"
        )

    actual_counts = {
        "equity": sum(t in SP500_EQUITIES for t in selected),
        "bond": sum(t in BOND_ETFS for t in selected),
        "etf": sum(t in BROAD_ETFS for t in selected),
        "crypto": sum(t in CRYPTOS for t in selected),
    }

    print(
        f"Stratified M={M}: "
        f"{actual_counts['equity']} equities, "
        f"{actual_counts['bond']} bonds, "
        f"{actual_counts['etf']} ETFs, "
        f"{actual_counts['crypto']} crypto"
    )

    return selected


# ============================================================
# CVaR HELPER
# ============================================================

def empirical_cvar(weights: np.ndarray, R_daily: np.ndarray, alpha: float = ALPHA) -> float:
    """
    Positive-loss historical CVaR.

    loss_t = -r_t^T w
    CVaR = average loss in the worst 1-alpha fraction of days.
    """
    losses = -R_daily @ weights
    cutoff = np.quantile(losses, alpha)
    tail_losses = losses[losses >= cutoff]

    if len(tail_losses) == 0:
        return float("nan")

    return float(np.mean(tail_losses))


# ============================================================
# GUROBI MODEL
# ============================================================

STATUS_MAP = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INTERRUPTED: "INTERRUPTED",
}


def safe_attr(model, attr, default=np.nan):
    try:
        return getattr(model, attr)
    except Exception:
        return default


def solve_epsilon_mip(
    assets: list,
    mu_annual: np.ndarray,
    R_daily: np.ndarray,
    epsilon: float,
):
    """
    Solve one epsilon-constrained mixed-integer CVaR portfolio problem.

    Maximise:
        annualised expected return

    Subject to:
        CVaR_0.95 <= epsilon,
        budget,
        cardinality,
        linkage bounds,
        equity sector caps,
        crypto cap,
        bond floor.

    This is a mixed-integer linear CVaR model, not MIQP, because CVaR is
    linearised using VaR and shortfall variables.
    """
    T, n = R_daily.shape

    model = gp.Model("epsilon_cvar_portfolio_scaling")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT_PER_EPS
    model.Params.MIPGap = MIP_GAP_TARGET

    # Decision variables
    w = model.addVars(n, lb=0.0, ub=W_MAX, vtype=GRB.CONTINUOUS, name="w")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    # CVaR auxiliary variables
    v = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="VaR")
    u = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="u")

    # Budget
    model.addConstr(
        gp.quicksum(w[i] for i in range(n)) == 1.0,
        name="budget"
    )

    # Cardinality
    model.addConstr(
        gp.quicksum(z[i] for i in range(n)) <= min(K, n),
        name="cardinality"
    )

    # Linkage constraints
    for i in range(n):
        model.addConstr(w[i] <= W_MAX * z[i], name=f"upper_link_{i}")
        model.addConstr(w[i] >= W_MIN * z[i], name=f"lower_link_{i}")

    # Bond floor
    bond_idx = [i for i, a in enumerate(assets) if a in BOND_ETFS]
    if BOND_FLOOR > 0:
        if len(bond_idx) == 0:
            model.addConstr(0 >= BOND_FLOOR, name="bond_floor_infeasible")
        else:
            model.addConstr(
                gp.quicksum(w[i] for i in bond_idx) >= BOND_FLOOR,
                name="bond_floor"
            )

    # Crypto cap
    crypto_idx = [i for i, a in enumerate(assets) if a in CRYPTOS]
    if len(crypto_idx) > 0:
        model.addConstr(
            gp.quicksum(w[i] for i in crypto_idx) <= CRYPTO_CAP,
            name="crypto_cap"
        )

    # Sector caps
    sector_to_indices = {}
    for i, asset in enumerate(assets):
        sector = SECTOR_MAP.get(asset, "Unknown")
        sector_to_indices.setdefault(sector, []).append(i)

    for sector, idxs in sector_to_indices.items():
        # Match the main Gurobi model:
        # sector caps apply to equity sectors only.
        # Bond, ETF, and Crypto have separate constraints.
        if sector in ["Bond", "ETF", "Crypto"]:
            continue

        model.addConstr(
            gp.quicksum(w[i] for i in idxs) <= SECTOR_CAP,
            name=f"sector_cap_{sector}"
        )

    # CVaR linearisation:
    # u_t >= loss_t - VaR
    # loss_t = -r_t^T w
    for t in range(T):
        daily_loss = -gp.quicksum(float(R_daily[t, i]) * w[i] for i in range(n))
        model.addConstr(
            u[t] >= daily_loss - v,
            name=f"shortfall_{t}"
        )

    cvar_expr = v + (1.0 / (T * (1.0 - ALPHA))) * gp.quicksum(u[t] for t in range(T))

    model.addConstr(
        cvar_expr <= epsilon,
        name="epsilon_cvar_limit"
    )

    # Objective: maximise annualised expected return
    model.setObjective(
        gp.quicksum(float(mu_annual[i]) * w[i] for i in range(n)),
        GRB.MAXIMIZE
    )

    model.optimize()

    status = STATUS_MAP.get(model.Status, str(model.Status))
    has_solution = model.SolCount > 0

    result = {
        "epsilon": float(epsilon),
        "status": status,
        "is_optimal": bool(model.Status == GRB.OPTIMAL),
        "has_solution": bool(has_solution),
        "runtime_sec": float(safe_attr(model, "Runtime")),
        "node_count": float(safe_attr(model, "NodeCount")),
        "mip_gap": float(safe_attr(model, "MIPGap")),
        "objective_return": np.nan,
        "realised_cvar": np.nan,
        "selected_count": np.nan,
        "selected_assets": "",
    }

    if has_solution:
        weights = np.array([w[i].X for i in range(n)])
        selected_assets = [assets[i] for i in range(n) if weights[i] > 1e-6]

        result.update({
            "objective_return": float(mu_annual @ weights),
            "realised_cvar": empirical_cvar(weights, R_daily, ALPHA),
            "selected_count": int(np.sum(weights > 1e-6)),
            "selected_assets": ",".join(selected_assets),
        })

    return result


# ============================================================
# MAIN SCALING EXPERIMENT
# ============================================================

def run_scaling_experiment():
    prices = load_prices()
    all_assets = get_asset_order(prices)

    print("\n=== Gurobi Full-Frontier Scaling Experiment ===")
    print(f"Available assets in prices.csv: {len(all_assets)}")
    print(f"Epsilon points per M: {len(EPS_GRID)}")
    print(f"Time limit per epsilon: {TIME_LIMIT_PER_EPS}s")
    print(f"MIP gap target: {MIP_GAP_TARGET}\n")

    all_rows = []

    for M in M_LIST:
        assets = select_assets_for_M(all_assets, M)

        if assets is None:
            print(f"[SKIP] M={M}: only {len(all_assets)} assets available in prices.csv.")
            continue

        print(f"\n--- Running M={M} with {len(assets)} assets ---")

        prices_subset = prices[assets]
        returns_df, mu_annual, R_daily = compute_returns(prices_subset)

        print(
            f"Aligned return matrix: T={R_daily.shape[0]} days, "
            f"M={R_daily.shape[1]} assets, "
            f"date range {returns_df.index.min().date()} to "
            f"{returns_df.index.max().date()}"
        )

        for j, eps in enumerate(EPS_GRID, start=1):
            print(f"  [{j:02d}/{len(EPS_GRID)}] epsilon={eps:.5f} ... ", end="")

            result = solve_epsilon_mip(
                assets=assets,
                mu_annual=mu_annual,
                R_daily=R_daily,
                epsilon=eps,
            )

            result["M"] = M
            result["T_days"] = R_daily.shape[0]
            result["n_assets_used_in_model"] = len(assets)

            all_rows.append(result)

            print(
                f"{result['status']}, "
                f"time={result['runtime_sec']:.2f}s, "
                f"gap={result['mip_gap']}, "
                f"nodes={result['node_count']:.0f}, "
                f"ret={result['objective_return']}"
            )

            # Save after every solve so progress is not lost.
            pd.DataFrame(all_rows).to_csv(
                OUT_DIR / "gurobi_scaling_full_frontier_raw.csv",
                index=False
            )

    raw_df = pd.DataFrame(all_rows)

    if raw_df.empty:
        raise RuntimeError("No scaling results produced.")

    raw_csv = OUT_DIR / "gurobi_scaling_full_frontier_raw.csv"
    summary_csv = OUT_DIR / "gurobi_scaling_full_frontier_summary.csv"
    json_file = OUT_DIR / "gurobi_scaling_full_frontier_summary.json"

    raw_df.to_csv(raw_csv, index=False)

    # For gap summaries, ignore infeasible rows where MIPGap can be inf.
    def finite_max_gap(x):
        x = pd.to_numeric(x, errors="coerce")
        x = x[np.isfinite(x)]
        return float(x.max()) if len(x) else np.nan

    def finite_median_gap(x):
        x = pd.to_numeric(x, errors="coerce")
        x = x[np.isfinite(x)]
        return float(x.median()) if len(x) else np.nan

    summary = (
        raw_df
        .groupby("M")
        .agg(
            epsilon_points=("epsilon", "count"),
            feasible_points=("has_solution", "sum"),
            optimal_points=("is_optimal", "sum"),
            total_runtime_sec=("runtime_sec", "sum"),
            median_runtime_sec=("runtime_sec", "median"),
            max_runtime_sec=("runtime_sec", "max"),
            total_nodes=("node_count", "sum"),
            median_nodes=("node_count", "median"),
            max_mip_gap=("mip_gap", finite_max_gap),
            median_mip_gap=("mip_gap", finite_median_gap),
            best_return=("objective_return", "max"),
            lowest_cvar=("realised_cvar", "min"),
        )
        .reset_index()
    )

    summary.to_csv(summary_csv, index=False)

    with open(json_file, "w") as f:
        json.dump(
            {
                "settings": {
                    "alpha": ALPHA,
                    "K": K,
                    "w_min": W_MIN,
                    "w_max": W_MAX,
                    "sector_cap": SECTOR_CAP,
                    "crypto_cap": CRYPTO_CAP,
                    "bond_floor": BOND_FLOOR,
                    "M_LIST": M_LIST,
                    "EPS_GRID": [float(x) for x in EPS_GRID],
                    "time_limit_per_epsilon": TIME_LIMIT_PER_EPS,
                    "mip_gap_target": MIP_GAP_TARGET,
                    "sampling": "stratified by asset class",
                },
                "summary": summary.to_dict(orient="records"),
            },
            f,
            indent=2
        )

    print("\n=== DONE ===")
    print(f"Raw results saved to:     {raw_csv}")
    print(f"Summary saved to:         {summary_csv}")
    print(f"JSON summary saved to:    {json_file}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_scaling_experiment()