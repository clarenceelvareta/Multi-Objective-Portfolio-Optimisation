import numpy as np
from pymoo.core.problem import ElementwiseProblem
from scipy.optimize import minimize
from compute_cvar import cvar
from config import (W_MIN, W_MAX, K, SECTOR_CAP, CRYPTO_CAP, 
                    BOND_FLOOR, BOND_ETFS, CRYPTOS, SECTOR_MAP)

class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, scenarios, tickers):
        self.mu = mu.to_numpy() if hasattr(mu, 'to_numpy') else mu
        self.scenarios = np.asarray(scenarios)
        self.tickers = tickers
        self.M = len(tickers)
        
        # Precompute indexes for constraints
        self.sector_idxs = {}
        for i, t in enumerate(tickers):
            sec = SECTOR_MAP.get(t, "Unknown")
            if sec not in self.sector_idxs:
                self.sector_idxs[sec] = []
            self.sector_idxs[sec].append(i)
            
        self.bond_idxs = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
        self.crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]

        # We tell pymoo we have M variables (the z binaries), 2 objectives, and 0 constraints
        # (We handle constraints internally via the repair function for simplicity in GA)
        super().__init__(n_var=self.M, n_obj=2, n_constr=0, xl=0, xu=1, vtype=int)

    def _evaluate(self, z, out, *args, **kwargs):
        z = z.astype(int)
        
        # 1. Get selected assets
        selected = np.where(z == 1)[0]
        
        # If GA mutated to < K assets, heavily penalize so it dies out
        if len(selected) < K or len(selected) == 0:
            out["F"] = [1.0, 1.0] # Worst possible values
            return

        # 2. Optimize continuous weights (w) for the selected K assets using Scipy
        mu_sub = self.mu[selected]
        scen_sub = self.scenarios[:, selected]
        
        # Initial guess: equal weight
        w0 = np.ones(len(selected)) / len(selected)
        
        # Bounds for the sub-problem
        bounds = [(W_MIN, W_MAX) for _ in range(len(selected))]

        def objective(w_sub):
            # Minimize negative return, and CVaR
            ret = -np.dot(mu_sub, w_sub)
            risk = cvar(w_sub, scen_sub)
            return [ret, risk]

        # Scipy minimize (using SLSQP which supports bounds)
        res = minimize(lambda w: objective(w)[0], w0, method='SLSQP', bounds=bounds)
        w_opt_sub = res.x

        # 3. Map weights back to full M-length array
        w_full = np.zeros(self.M)
        w_full[selected] = w_opt_sub

        # 4. Check structural constraints (Bond floor, Crypto cap)
        # If violated, apply a penalty to the objectives so NSGA-II rejects it
        penalty = 0.0
        
        # Bond floor check
        if self.bond_idxs:
            bond_w = w_full[self.bond_idxs].sum()
            if bond_w < BOND_FLOOR:
                penalty += (BOND_FLOOR - bond_w) * 10 # Heavy penalty
                
        # Crypto cap check
        if self.crypto_idxs:
            crypto_w = w_full[self.crypto_idxs].sum()
            if crypto_w > CRYPTO_CAP:
                penalty += (crypto_w - CRYPTO_CAP) * 10

        # 5. Calculate final objectives
        final_ret = -np.dot(self.mu, w_full) # Negative because pymoo minimizes
        final_cvar = cvar(w_full, self.scenarios)
        
        out["F"] = [final_ret + penalty, final_cvar + penalty]