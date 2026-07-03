# gurobi_optimizer.py
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
from config import (W_MIN, W_MAX, K, SECTOR_CAP, CRYPTO_CAP,
                    BOND_FLOOR, CVAR_ALPHA, BOND_ETFS, CRYPTOS, SECTOR_MAP)


def _build_cvar_model(m, mu_arr, scenarios_arr, tickers,
                       sector_map_inv, bond_idxs, crypto_idxs,
                       eps, cvar_denom):
    """
    Shared helper — builds one Gurobi model for a given epsilon.
    Called by both optimize_gurobi and scaling_experiment_proper.
    """
    M = len(mu_arr)
    T = len(scenarios_arr)

    w   = m.addMVar(M, lb=0.0, ub=W_MAX,  name="w")
    z   = m.addMVar(M, vtype=GRB.BINARY,   name="z")
    VaR = m.addMVar(1, lb=-GRB.INFINITY,   name="VaR")  # MVar(1) not addVar
    U   = m.addMVar(T, lb=0.0,             name="U")

    # ── standard constraints ───────────────────────────────────────────
    m.addConstr(w.sum() == 1.0,  "Budget")
    m.addConstr(z.sum() <= K,    "Cardinality")
    m.addConstr(w >= W_MIN * z,  "Link_Lower")
    m.addConstr(w <= W_MAX * z,  "Link_Upper")

    for sec, idxs in sector_map_inv.items():
        if sec not in ["Bond", "ETF", "Crypto"]:
            m.addConstr(
                gp.quicksum(w[i] for i in idxs) <= SECTOR_CAP,
                f"SecCap_{sec}"
            )
    if crypto_idxs:
        m.addConstr(
            gp.quicksum(w[i] for i in crypto_idxs) <= CRYPTO_CAP,
            "Crypto_Cap"
        )
    if bond_idxs:
        m.addConstr(
            gp.quicksum(w[i] for i in bond_idxs) >= BOND_FLOOR,
            "Bond_Floor"
        )

    # ── CVaR linearisation ─────────────────────────────────────────────
    # U_t >= -r_t^T w - VaR  for all t
    # KEY FIX: use np.ones((T,1)) @ VaR to broadcast MVar(1) across T rows
    ones_col = np.ones((T, 1))
    m.addConstr(
        U >= -scenarios_arr @ w - ones_col @ VaR,
        "Shortfall"
    )
    m.addConstr(
        VaR.sum() + (1.0 / cvar_denom) * U.sum() <= eps,
        "Epsilon_CVaR"
    )

    m.setObjective(mu_arr @ w, GRB.MAXIMIZE)

    return w, z, VaR, U


def optimize_gurobi(mu, scenarios, tickers, epsilon_values=None, verbose=False):
    M             = len(tickers)
    T             = len(scenarios)
    alpha         = CVAR_ALPHA
    cvar_denom    = T * (1.0 - alpha)
    mu_arr        = mu.to_numpy() if hasattr(mu, "to_numpy") else np.asarray(mu)
    scenarios_arr = np.asarray(scenarios, dtype=float)

    if epsilon_values is None:
        epsilon_values = np.linspace(0.010, 0.045, 20)

    # precompute index maps
    sector_map_inv = {}
    for idx, ticker in enumerate(tickers):
        sec = SECTOR_MAP.get(ticker, "Unknown")
        sector_map_inv.setdefault(sec, []).append(idx)

    bond_idxs   = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
    crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]

    fronts     = []
    total_time = 0.0

    for eps in epsilon_values:
        t0 = time.time()
        m  = gp.Model("CVaR_Pareto")
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", 120)

        w, z, VaR, U = _build_cvar_model(
            m, mu_arr, scenarios_arr, tickers,
            sector_map_inv, bond_idxs, crypto_idxs,
            eps, cvar_denom
        )

        try:
            m.optimize()
            elapsed     = time.time() - t0
            total_time += elapsed

            if m.Status == GRB.OPTIMAL:
                w_opt         = np.array(w.X)
                z_opt         = np.array(z.X)
                var_val       = float(VaR.X[0])
                realised_cvar = var_val + (1.0 / cvar_denom) * np.array(U.X).sum()
                realised_ret  = float(mu_arr @ w_opt)

                fronts.append({
                    "w":    w_opt,
                    "z":    z_opt,
                    "ret":  realised_ret,
                    "cvar": realised_cvar,
                })
                if verbose:
                    print(f"  eps={eps:.4f}  ret={realised_ret:.4f}  "
                          f"cvar={realised_cvar:.4f}  t={elapsed:.1f}s")
            else:
                if verbose:
                    print(f"  eps={eps:.4f}  status={m.Status} (infeasible)")

        except Exception as e:
            if verbose:
                print(f"  eps={eps:.4f}  Error: {e}")

    return fronts, total_time


def scaling_experiment_proper(mu, scenarios, tickers,
                               n_epsilons=20, time_limit=300):
    M_values      = [20, 40, 60, 80, 100]
    results       = {}
    alpha         = CVAR_ALPHA
    mu_arr        = mu.to_numpy() if hasattr(mu, "to_numpy") else np.asarray(mu)
    scenarios_arr = np.asarray(scenarios, dtype=float)

    bond_idxs   = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
    crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]
    other_idxs  = [i for i in range(len(tickers))
                   if i not in bond_idxs and i not in crypto_idxs]

    for M_size in M_values:
        print(f"\nM={M_size} — sweeping {n_epsilons} epsilon points...")

        # ── stratified subset ──────────────────────────────────────────
        n_bonds  = max(2, round(len(bond_idxs)   / len(tickers) * M_size))
        n_crypto = max(1, round(len(crypto_idxs) / len(tickers) * M_size))
        n_other  = M_size - n_bonds - n_crypto

        rng   = np.random.default_rng(42)
        idx_b = rng.choice(bond_idxs,   min(n_bonds,  len(bond_idxs)),   replace=False)
        idx_c = rng.choice(crypto_idxs, min(n_crypto, len(crypto_idxs)), replace=False)
        idx_o = rng.choice(other_idxs,  min(n_other,  len(other_idxs)),  replace=False)
        sel   = np.concatenate([idx_b, idx_c, idx_o]).astype(int)

        sub_mu        = mu_arr[sel]
        sub_scenarios = scenarios_arr[:, sel]
        sub_tickers   = [tickers[i] for i in sel]
        T, M_sub      = sub_scenarios.shape
        cvar_denom    = T * (1.0 - alpha)

        # ── adaptive epsilon range ─────────────────────────────────────
        w_eq     = np.ones(M_sub) / M_sub
        port_ret = sub_scenarios @ w_eq
        cvar_min = float(-np.quantile(port_ret, 1 - alpha) * 0.5)
        cvar_max = float(-np.quantile(port_ret, 1 - alpha) * 2.0)
        eps_list = np.linspace(cvar_min, cvar_max, n_epsilons)

        # ── precompute index maps for THIS subset ──────────────────────
        sub_sector_map  = {}
        for i, t in enumerate(sub_tickers):
            sec = SECTOR_MAP.get(t, "Unknown")
            sub_sector_map.setdefault(sec, []).append(i)
        sub_bond_idxs   = [i for i, t in enumerate(sub_tickers) if t in BOND_ETFS]
        sub_crypto_idxs = [i for i, t in enumerate(sub_tickers) if t in CRYPTOS]

        t0     = time.time()
        solved = 0

        for eps in eps_list:
            try:
                model = gp.Model()
                model.setParam("OutputFlag", 0)
                model.setParam("TimeLimit", time_limit / n_epsilons)

                # ── build model manually (not via _build_cvar_model) ───
                # so we can control K_sub independently
                K_sub    = min(K, M_sub)
                ones_col = np.ones((T, 1))

                w   = model.addMVar(M_sub, lb=0.0, ub=W_MAX, name="w")
                z   = model.addMVar(M_sub, vtype=GRB.BINARY, name="z")
                VaR = model.addMVar(1, lb=-GRB.INFINITY,      name="VaR")
                U   = model.addMVar(T, lb=0.0,                name="U")

                model.addConstr(w.sum() == 1.0,    "Budget")
                model.addConstr(z.sum() <= K_sub,  "Cardinality")
                model.addConstr(w >= W_MIN * z,    "Link_Lower")
                model.addConstr(w <= W_MAX * z,    "Link_Upper")

                for sec, idxs in sub_sector_map.items():
                    if sec not in ["Bond", "ETF", "Crypto"]:
                        model.addConstr(
                            gp.quicksum(w[i] for i in idxs) <= SECTOR_CAP,
                            f"SecCap_{sec}"
                        )
                if sub_crypto_idxs:
                    model.addConstr(
                        gp.quicksum(w[i] for i in sub_crypto_idxs) <= CRYPTO_CAP,
                        "Crypto_Cap"
                    )
                if sub_bond_idxs:
                    model.addConstr(
                        gp.quicksum(w[i] for i in sub_bond_idxs) >= BOND_FLOOR,
                        "Bond_Floor"
                    )

                # ── CVaR with correct MVar broadcasting ────────────────
                model.addConstr(
                    U >= -sub_scenarios @ w - ones_col @ VaR,
                    "Shortfall"
                )
                model.addConstr(
                    VaR.sum() + (1.0 / cvar_denom) * U.sum() <= eps,
                    "Epsilon_CVaR"
                )
                model.setObjective(sub_mu @ w, GRB.MAXIMIZE)
                model.optimize()

                if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                    solved += 1

            except Exception as e:
                print(f"  Warning at eps={eps:.4f}: {e}")

        elapsed         = time.time() - t0
        results[M_size] = elapsed
        print(f"  -> {elapsed:.2f}s  ({solved}/{n_epsilons} solved)")

    return results