import numpy as np
from pymoo.core.problem import ElementwiseProblem
from scipy.optimize import minimize
from compute_cvar import cvar
from config import (W_MIN, W_MAX, K, SECTOR_CAP, CRYPTO_CAP, 
                    BOND_FLOOR, BOND_ETFS, CRYPTOS, SECTOR_MAP)


class PortfolioProblem(ElementwiseProblem):
    """
    CONSTRAINT HANDLING 1: REPAIR METHOD
    The GA evolves binary z variables. If it creates an invalid portfolio,
    we rely on the repair() function in constraints.py to fix it before evaluating.
    """
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

        # M binary variables (the z selection vector)
        super().__init__(n_var=self.M, n_obj=2, n_constr=0, xl=0, xu=1, vtype=int)

    def _evaluate(self, z, out, *args, **kwargs):
        z = z.astype(int)
        selected = np.where(z == 1)[0]
        
        # If GA mutated to < K or > K assets, heavily penalize
        if len(selected) < K or len(selected) == 0:
            out["F"] = [1.0, 1.0] 
            return

        # Use Local Search (SLSQP) to find exact optimal weights for selected assets
        mu_sub = self.mu[selected]
        scen_sub = self.scenarios[:, selected]
        w0 = np.ones(len(selected)) / len(selected)
        bounds = [(W_MIN, W_MAX) for _ in range(len(selected))]

        def objective(w_sub):
            ret = -np.dot(mu_sub, w_sub)
            risk = cvar(w_sub, scen_sub)
            return [ret, risk]

        # Scipy minimizes the first objective (Return)
        res = minimize(lambda w: objective(w)[0], w0, method='SLSQP', bounds=bounds)
        w_opt_sub = res.x

        # Map weights back to full M-length array
        w_full = np.zeros(self.M)
        w_full[selected] = w_opt_sub

        # Check structural constraints (Bond floor, Crypto cap)
        penalty = 0.0
        
        if self.bond_idxs:
            bond_w = w_full[self.bond_idxs].sum()
            if bond_w < BOND_FLOOR:
                penalty += (BOND_FLOOR - bond_w) * 10
                
        if self.crypto_idxs:
            crypto_w = w_full[self.crypto_idxs].sum()
            if crypto_w > CRYPTO_CAP:
                penalty += (crypto_w - CRYPTO_CAP) * 10

        # Calculate final objectives
        final_ret = -np.dot(self.mu, w_full)
        final_cvar = cvar(w_full, self.scenarios)
        
        out["F"] = [final_ret + penalty, final_cvar + penalty]


class PortfolioProblemPenalty(ElementwiseProblem):
    """
    CONSTRAINT HANDLING 2: PENALTY METHOD
    The GA evolves binary z variables. We DO NOT repair them. Instead, if 
    constraints are violated, we add a massive mathematical penalty to their
    objective scores so they lose the evolutionary race.
    """
    def __init__(self, mu, scenarios, tickers):
        self.mu = mu.to_numpy() if hasattr(mu, 'to_numpy') else mu
        self.scenarios = np.asarray(scenarios)
        self.tickers = tickers
        self.M = len(tickers)
        
        self.sector_idxs = {}
        for i, t in enumerate(tickers):
            sec = SECTOR_MAP.get(t, "Unknown")
            if sec not in self.sector_idxs:
                self.sector_idxs[sec] = []
            self.sector_idxs[sec].append(i)
            
        self.bond_idxs = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
        self.crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]

        super().__init__(n_var=self.M, n_obj=2, n_constr=0, xl=0, xu=1, vtype=int)

    def _evaluate(self, z, out, *args, **kwargs):
        z = z.astype(int)
        selected = np.where(z == 1)[0]
        
        # Death penalty if it doesn't pick exactly K assets
        if len(selected) != K or len(selected) == 0:
            out["F"] = [1.0, 1.0] 
            return

        # Find continuous weights using Scipy (ONLY MINIMIZE RETURN - FIXED BUG)
        mu_sub = self.mu[selected]
        scen_sub = self.scenarios[:, selected]
        w0 = np.ones(len(selected)) / len(selected)
        bounds = [(W_MIN, W_MAX) for _ in range(len(selected))]

        res = minimize(lambda w: -np.dot(mu_sub, w), 
                       w0, method='SLSQP', bounds=bounds)
        w_opt_sub = res.x

        # Map to full array
        w_full = np.zeros(self.M)
        w_full[selected] = w_opt_sub

        # Calculate raw objectives
        raw_ret = -np.dot(self.mu, w_full)
        raw_cvar = cvar(w_full, self.scenarios)

        # CALCULATE PENALTIES (No repairing!)
        penalty = 0.0
        
        # Penalty 1: Bond Floor
        if self.bond_idxs:
            bond_w = w_full[self.bond_idxs].sum()
            if bond_w < BOND_FLOOR:
                penalty += (BOND_FLOOR - bond_w) * 50.0 # Heavier multiplier than repair
                
        # Penalty 2: Crypto Cap
        if self.crypto_idxs:
            crypto_w = w_full[self.crypto_idxs].sum()
            if crypto_w > CRYPTO_CAP:
                penalty += (crypto_w - CRYPTO_CAP) * 50.0
                
        # Penalty 3: Sector Caps
        for sec, idxs in self.sector_idxs.items():
            if sec not in ["Bond", "ETF", "Crypto"]:
                sec_w = w_full[idxs].sum()
                if sec_w > SECTOR_CAP:
                    penalty += (sec_w - SECTOR_CAP) * 50.0

        # Apply penalty to objectives
        out["F"] = [raw_ret + penalty, raw_cvar + penalty]


class PortfolioProblemDecoder(ElementwiseProblem):
    """
    CONSTRAINT HANDLING 3: PRESERVING STRATEGY (DECODER)
    Instead of binary variables, the GA evolves continuous "priority scores" [0, 1].
    A decoder function automatically picks the Top K assets. This makes it
    mathematically IMPOSSIBLE to violate the cardinality constraint.
    """
    def __init__(self, mu, scenarios, tickers):
        self.mu = mu.to_numpy() if hasattr(mu, 'to_numpy') else mu
        self.scenarios = np.asarray(scenarios)
        self.tickers = tickers
        self.M = len(tickers)
        
        # Variables are continuous floats between 0 and 1, not binaries!
        super().__init__(n_var=self.M, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # THE DECODER: Sort assets by priority score, pick Top K
        selected = np.argsort(x)[-K:] 
        
        # Use Local Search (SLSQP) to find exact weights for these K assets
        mu_sub = self.mu[selected]
        scen_sub = self.scenarios[:, selected]
        w0 = np.ones(K) / K
        bounds = [(W_MIN, W_MAX) for _ in range(K)]

        # Scipy minimizes the first objective (Return)
        res = minimize(lambda w: -np.dot(mu_sub, w), 
                       w0, method='SLSQP', bounds=bounds)
        w_opt = res.x

        # Return objectives
        out["F"] = [-np.dot(mu_sub, w_opt), cvar(w_opt, scen_sub)]