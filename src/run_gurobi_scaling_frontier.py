from pathlib import Path
import json
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from config import (
    ALL_TICKERS,
    BOND_ETFS,
    BOND_FLOOR,
    CRYPTO_CAP,
    CRYPTOS,
    CVAR_ALPHA,
    K,
    SECTOR_CAP,
    SECTOR_MAP,
    W_MAX,
    W_MIN,
)


ALPHA = CVAR_ALPHA
M_LIST = [20, 40, 60, 80, 100, 150, 200]
EPS_GRID = np.linspace(0.010, 0.045, 20)
TIME_LIMIT_PER_EPS = 300
MIP_GAP_TARGET = 1e-4

ROOT = Path(__file__).resolve().parents[1]
PRICE_FILE = ROOT / "src" / "raw" / "prices.csv"
OUT_DIR = ROOT / "scaling_results"

BOND_ASSETS = set(BOND_ETFS)
CRYPTO_ASSETS = set(CRYPTOS)

STATUS_MAP = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INTERRUPTED: "INTERRUPTED",
}


def load_prices(price_file=PRICE_FILE):
    if not price_file.exists():
        raise FileNotFoundError(
            f"Could not find {price_file}. Make sure src/raw/prices.csv exists."
        )

    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    prices.columns = [str(c).strip() for c in prices.columns]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices.dropna(axis=1, how="all")


def get_asset_order(prices):
    available = list(prices.columns)
    ordered = [ticker for ticker in ALL_TICKERS if ticker in available]
    leftovers = [ticker for ticker in available if ticker not in ordered]
    return ordered + leftovers


def select_assets_for_m(all_assets, m_target):
    """
    Prefix selection keeps the scaling experiment reproducible while forcing
    a bond into each sub-universe when the bond floor is active.
    """
    if m_target > len(all_assets):
        return None

    selected = list(all_assets[:m_target])
    if BOND_FLOOR > 0 and not any(ticker in BOND_ASSETS for ticker in selected):
        first_bond = next((ticker for ticker in all_assets if ticker in BOND_ASSETS), None)
        if first_bond is None:
            raise ValueError("Bond floor is active, but no bond asset was found.")
        selected[-1] = first_bond

    selected = list(dict.fromkeys(selected))
    for ticker in all_assets:
        if len(selected) >= m_target:
            break
        if ticker not in selected:
            selected.append(ticker)

    return selected


def compute_returns(prices_subset):
    clean_prices = prices_subset.ffill().dropna(axis=0, how="any")
    returns = np.log(clean_prices / clean_prices.shift(1)).dropna(axis=0, how="any")
    if returns.empty:
        raise ValueError("Return matrix is empty after alignment.")

    mu_annual = 252 * returns.mean().values
    return returns, mu_annual, returns.values


def empirical_cvar(weights, daily_returns, alpha=ALPHA):
    losses = -daily_returns @ weights
    cutoff = np.quantile(losses, alpha)
    tail_losses = losses[losses >= cutoff]
    if len(tail_losses) == 0:
        return float("nan")
    return float(np.mean(tail_losses))


def safe_attr(model, attr, default=np.nan):
    try:
        return getattr(model, attr)
    except Exception:
        return default


def solve_epsilon_mip(assets, mu_annual, daily_returns, epsilon):
    """
    Maximise annualised expected return subject to an empirical CVaR ceiling
    and the project portfolio constraints.
    """
    t_count, n_assets = daily_returns.shape

    model = gp.Model("epsilon_cvar_portfolio_scaling")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT_PER_EPS
    model.Params.MIPGap = MIP_GAP_TARGET

    w = model.addVars(n_assets, lb=0.0, ub=W_MAX, vtype=GRB.CONTINUOUS, name="w")
    z = model.addVars(n_assets, vtype=GRB.BINARY, name="z")
    var = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="VaR")
    shortfall = model.addVars(t_count, lb=0.0, vtype=GRB.CONTINUOUS, name="u")

    model.addConstr(gp.quicksum(w[i] for i in range(n_assets)) == 1.0, name="budget")
    model.addConstr(
        gp.quicksum(z[i] for i in range(n_assets)) <= min(K, n_assets),
        name="cardinality",
    )

    for i in range(n_assets):
        model.addConstr(w[i] <= W_MAX * z[i], name=f"upper_link_{i}")
        model.addConstr(w[i] >= W_MIN * z[i], name=f"lower_link_{i}")

    bond_idx = [i for i, ticker in enumerate(assets) if ticker in BOND_ASSETS]
    if BOND_FLOOR > 0:
        if bond_idx:
            model.addConstr(
                gp.quicksum(w[i] for i in bond_idx) >= BOND_FLOOR,
                name="bond_floor",
            )
        else:
            model.addConstr(0 >= BOND_FLOOR, name="bond_floor_infeasible")

    crypto_idx = [i for i, ticker in enumerate(assets) if ticker in CRYPTO_ASSETS]
    if crypto_idx:
        model.addConstr(
            gp.quicksum(w[i] for i in crypto_idx) <= CRYPTO_CAP,
            name="crypto_cap",
        )

    sector_to_indices = {}
    for i, ticker in enumerate(assets):
        sector = SECTOR_MAP.get(ticker)
        if sector in (None, "Bond", "ETF", "Crypto"):
            continue
        sector_to_indices.setdefault(sector, []).append(i)

    for sector, idxs in sector_to_indices.items():
        model.addConstr(
            gp.quicksum(w[i] for i in idxs) <= SECTOR_CAP,
            name=f"sector_cap_{sector}",
        )

    for t in range(t_count):
        daily_loss = -gp.quicksum(
            float(daily_returns[t, i]) * w[i] for i in range(n_assets)
        )
        model.addConstr(shortfall[t] >= daily_loss - var, name=f"shortfall_{t}")

    cvar_expr = var + (1.0 / (t_count * (1.0 - ALPHA))) * gp.quicksum(
        shortfall[t] for t in range(t_count)
    )
    model.addConstr(cvar_expr <= epsilon, name="epsilon_cvar_limit")

    model.setObjective(
        gp.quicksum(float(mu_annual[i]) * w[i] for i in range(n_assets)),
        GRB.MAXIMIZE,
    )
    model.optimize()

    status = STATUS_MAP.get(model.Status, str(model.Status))
    has_solution = model.SolCount > 0
    result = {
        "epsilon": float(epsilon),
        "status": status,
        "is_optimal": model.Status == GRB.OPTIMAL,
        "has_solution": has_solution,
        "runtime_sec": float(safe_attr(model, "Runtime")),
        "node_count": float(safe_attr(model, "NodeCount")),
        "mip_gap": float(safe_attr(model, "MIPGap")),
        "objective_return": np.nan,
        "realised_cvar": np.nan,
        "selected_count": np.nan,
        "selected_assets": "",
    }

    if has_solution:
        weights = np.array([w[i].X for i in range(n_assets)])
        selected_assets = [assets[i] for i in range(n_assets) if weights[i] > 1e-6]
        result.update(
            {
                "objective_return": float(mu_annual @ weights),
                "realised_cvar": empirical_cvar(weights, daily_returns),
                "selected_count": int(np.sum(weights > 1e-6)),
                "selected_assets": ",".join(selected_assets),
            }
        )

    return result


def run_scaling_experiment():
    OUT_DIR.mkdir(exist_ok=True)
    prices = load_prices()
    all_assets = get_asset_order(prices)

    print("\n=== Gurobi Full-Frontier Scaling Experiment ===")
    print(f"Available assets in prices.csv: {len(all_assets)}")
    print(f"Bond assets detected: {sorted(BOND_ASSETS)}")
    print(f"Crypto assets detected: {sorted(CRYPTO_ASSETS)}")
    print(f"Epsilon points per M: {len(EPS_GRID)}")
    print(f"Time limit per epsilon: {TIME_LIMIT_PER_EPS}s")

    all_rows = []
    raw_csv = OUT_DIR / "gurobi_scaling_full_frontier_raw.csv"
    summary_csv = OUT_DIR / "gurobi_scaling_full_frontier_summary.csv"
    json_file = OUT_DIR / "gurobi_scaling_full_frontier_summary.json"

    for m_target in M_LIST:
        assets = select_assets_for_m(all_assets, m_target)
        if assets is None:
            print(
                f"[SKIP] M={m_target}: only {len(all_assets)} assets available in prices.csv."
            )
            continue

        print(f"\n--- Running M={m_target} with {len(assets)} assets ---")
        returns_df, mu_annual, daily_returns = compute_returns(prices[assets])
        print(
            f"Aligned return matrix: T={daily_returns.shape[0]} days, "
            f"M={daily_returns.shape[1]} assets, "
            f"date range {returns_df.index.min().date()} to "
            f"{returns_df.index.max().date()}"
        )

        for idx, epsilon in enumerate(EPS_GRID, start=1):
            print(f"  [{idx:02d}/{len(EPS_GRID)}] epsilon={epsilon:.5f} ... ", end="")
            result = solve_epsilon_mip(assets, mu_annual, daily_returns, epsilon)
            result["M"] = m_target
            result["T_days"] = int(daily_returns.shape[0])
            result["n_assets_used_in_model"] = int(len(assets))
            all_rows.append(result)

            print(
                f"{result['status']}, "
                f"time={result['runtime_sec']:.2f}s, "
                f"gap={result['mip_gap']}, "
                f"nodes={result['node_count']:.0f}, "
                f"ret={result['objective_return']}"
            )

            pd.DataFrame(all_rows).to_csv(raw_csv, index=False)

    raw_df = pd.DataFrame(all_rows)
    if raw_df.empty:
        raise RuntimeError("No scaling results produced.")

    raw_df.to_csv(raw_csv, index=False)
    summary = (
        raw_df.groupby("M")
        .agg(
            epsilon_points=("epsilon", "count"),
            feasible_points=("has_solution", "sum"),
            optimal_points=("is_optimal", "sum"),
            total_runtime_sec=("runtime_sec", "sum"),
            median_runtime_sec=("runtime_sec", "median"),
            max_runtime_sec=("runtime_sec", "max"),
            total_nodes=("node_count", "sum"),
            median_nodes=("node_count", "median"),
            max_mip_gap=("mip_gap", "max"),
            median_mip_gap=("mip_gap", "median"),
            best_return=("objective_return", "max"),
            lowest_cvar=("realised_cvar", "min"),
        )
        .reset_index()
    )
    summary.to_csv(summary_csv, index=False)

    settings = {
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
    }
    summary_records = json.loads(summary.to_json(orient="records"))
    payload = {"settings": settings, "summary": summary_records}
    with open(json_file, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== DONE ===")
    print(f"Raw results saved to:     {raw_csv}")
    print(f"Summary saved to:         {summary_csv}")
    print(f"JSON summary saved to:    {json_file}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    return {
        **payload,
        "output_files": {
            "raw_csv": str(raw_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(json_file),
        },
    }


if __name__ == "__main__":
    run_scaling_experiment()
