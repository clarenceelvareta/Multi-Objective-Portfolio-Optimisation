import numpy as np
import gurobipy as gp
from gurobipy import GRB
from config import (W_MIN, W_MAX, K, SECTOR_CAP, CRYPTO_CAP, 
                    BOND_FLOOR, CVAR_ALPHA, BOND_ETFS, CRYPTOS, SECTOR_MAP)

def optimize_gurobi(mu, scenarios, tickers, epsilon_values=None, verbose=False):
    """
    Solves the bi-objective CVaR-Return problem exactly using Gurobi.
    Uses epsilon-constraint method on CVaR to find the Pareto front.
    
    Returns:
        fronts: list of dicts, each containing 'w', 'z', 'ret', 'cvar'
        solve_time: total time taken
    """
    M = len(tickers)
    T = len(scenarios)
    alpha = CVAR_ALPHA
    n_scenarios = int(np.ceil(T * (1 - alpha))) # Number of tail scenarios
    
    # If no epsilon values provided, create a sweep of 15 points
    if epsilon_values is None:
        eps_min = 0.01
        eps_max = 0.08 # Adjust this based on your equal-weight CVaR (~0.023)
        epsilon_values = np.linspace(eps_min, eps_max, 15)

    fronts = []
    
    # Precompute mappings for constraints
    sector_map_inv = {}
    for idx, ticker in enumerate(tickers):
        sec = SECTOR_MAP.get(ticker, "Unknown")
        if sec not in sector_map_inv:
            sector_map_inv[sec] = []
        sector_map_inv[sec].append(idx)
        
    bond_idxs = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
    crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]

    for eps in epsilon_values:
        m = gp.Model("CVaR_Pareto")
        if not verbose:
            m.setParam('OutputFlag', 0)

        # --- Decision Variables ---
        w = m.addMVar(M, lb=0, ub=W_MAX, name="w")
        z = m.addMVar(M, vtype=GRB.BINARY, name="z")
        
        # Auxiliary variables for CVaR linearization (Rockafellar & Uryasev)
        VaR = m.addVar(lb=-GRB.INFINITY, name="VaR")
        U = m.addMVar(T, lb=0, name="U") # Shortfall variables

        # --- Constraints ---
        # 1. Budget
        m.addConstr(w.sum() == 1.0, "Budget")
        
        # 2. Cardinality
        m.addConstr(z.sum() <= K, "Cardinality")
        
        # 3. Linkage: W_MIN * z <= w <= W_MAX * z
        m.addConstr(w >= W_MIN * z, "Link_Lower")
        m.addConstr(w <= W_MAX * z, "Link_Upper")
        
        # 4. Sector Caps
        for sec, idxs in sector_map_inv.items():
            if sec not in ["Bond", "ETF", "Crypto"]: # Only apply to GICS sectors
                m.addConstr(w[idxs].sum() <= SECTOR_CAP, f"SecCap_{sec}")
                
        # 5. Crypto Cap
        if crypto_idxs:
            m.addConstr(w[crypto_idxs].sum() <= CRYPTO_CAP, "Crypto_Cap")
            
        # 6. Bond Floor
        if bond_idxs:
            m.addConstr(w[bond_idxs].sum() >= BOND_FLOOR, "Bond_Floor")

        # 7. CVaR Linearization Constraints
        # U_t >= -r_t^T w - VaR  ==>  Loss - VaR
        # Note: scenarios is TxM, so -scenarios @ w gives losses
        m.addConstr(U >= -scenarios @ w - VaR, "Shortfall_Def")
        
        # 8. Epsilon constraint on CVaR (This traces the Pareto front)
        # CVaR = VaR + (1/(T*(1-alpha))) * sum(U)
        m.addConstr(VaR + (1.0 / n_scenarios) * U.sum() <= eps, "Epsilon_CVaR")

        # --- Objective ---
        # Maximize Expected Return: mu^T w
        m.setObjective(mu.to_numpy() @ w, GRB.MAXIMIZE)

        # --- Solve ---
        try:
            m.optimize()
            
            if m.status == GRB.OPTIMAL:
                w_opt = w.X
                z_opt = (w_opt > 1e-4).astype(int) # Clean up floating point binaries
                
                # Calculate actual realized CVaR for this portfolio
                realized_cvar = VaR.X + (1.0 / n_scenarios) * U.X.sum()
                realized_ret = mu @ w_opt
                
                fronts.append({
                    'w': w_opt,
                    'z': z_opt,
                    'ret': realized_ret,
                    'cvar': realized_cvar
                })
        except gp.GurobiError as e:
            if verbose:
                print(f"Infeasible at epsilon {eps:.4f}: {e}")
            pass

    solve_time = m.Runtime if 'm' in locals() else 0
    return fronts, solve_time

def scaling_experiment(mu_full, scenarios_full, tickers_full):
    """
    Runs the Gurobi optimizer for M = 20, 40, 60, 80, 100 and records time.
    """
    sizes = [20, 40, 60, 80, 100]
    results = {}
    
    for M_target in sizes:
        print(f"Running Gurobi for M={M_target}...")
        # Take the first M_target assets to keep it simple
        mu_sub = mu_full[:M_target]
        scen_sub = scenarios_full[:, :M_target]
        tick_sub = tickers_full[:M_target]
        
        # Only ask for 1 point on the frontier to test pure solve speed
        fronts, time_taken = optimize_gurobi(mu_sub, scen_sub, tick_sub, epsilon_values=[0.05])
        results[M_target] = time_taken
        print(f"  -> M={M_target} solved in {time_taken:.2f} seconds ({len(fronts)} points)")
        
    return results