"""
pymoo problem definitions for the portfolio metaheuristics.

The pipeline uses pymoo algorithms (NSGA-II, MOEA/D, AGE-MOEA) over
portfolio selection encodings. Each candidate first determines a selected
asset set, then a local SLSQP solve assigns continuous weights for that
selected set. The returned objective vector follows pymoo's minimisation
convention:

* objective 1 = negative annualised return, so minimising improves return,
* objective 2 = positive CVaR loss, so minimising reduces tail risk.

Three subclasses implement the constraint-handling comparison:
Repair, Penalty, and Decoder.
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from scipy.optimize import minimize
from compute_cvar import cvar
from config import (W_MIN, W_MAX, K, SECTOR_CAP, CRYPTO_CAP,
                    BOND_FLOOR, BOND_ETFS, CRYPTOS, SECTOR_MAP)
from constraints import repair


class BasePortfolioProblem(ElementwiseProblem):
    """
    Base class shared by all three constraint-handling strategies.
    Centralises index mapping and SLSQP weight solver.
    """
    def __init__(self, mu, scenarios, tickers, vtype):
        self.mu        = mu.to_numpy() if hasattr(mu, "to_numpy") else np.asarray(mu)
        self.scenarios = np.asarray(scenarios)
        self.tickers   = tickers
        self.M         = len(tickers)

        # precompute index maps once
        self.sector_idxs = {}
        for i, t in enumerate(tickers):
            sec = SECTOR_MAP.get(t, "Unknown")
            self.sector_idxs.setdefault(sec, []).append(i)

        self.bond_idxs   = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
        self.crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]

        super().__init__(
            n_var=self.M, n_obj=2, n_constr=0,
            xl=0.0, xu=1.0, vtype=vtype
        )

    def _solve_weights(self, selected):
        """
        SLSQP local search: finds optimal continuous weights for
        a fixed set of selected assets.
        Returns w of length len(selected), summing to 1.
        """
        k      = len(selected)
        mu_sub = self.mu[selected]
        sc_sub = self.scenarios[:, selected]

        w0     = np.ones(k) / k
        bounds = [(W_MIN, W_MAX)] * k

        def objective(w):
            ret  = -np.dot(mu_sub, w)
            risk =  cvar(w, sc_sub)
            return 0.5 * ret + 0.25 * risk

        res = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"ftol": 1e-8, "maxiter": 150}
        )

        w_opt = np.clip(res.x, W_MIN, W_MAX)
        w_opt /= w_opt.sum()
        return w_opt

    def _apply_portfolio_constraints(self, w_full, selected):
        """
        Post-SLSQP enforcement of bond floor, crypto cap, and sector cap.
        Modifies w_full in-place and re-normalises.
        Only called by Decoder — Repair uses constraints.repair() instead.
        """
        # bond floor
        bond_sel = [i for i in self.bond_idxs if i in selected]
        if bond_sel:
            bw = w_full[bond_sel].sum()
            if bw < BOND_FLOOR:
                deficit  = BOND_FLOOR - bw
                per_bond = deficit / len(bond_sel)
                for i in bond_sel:
                    w_full[i] = min(w_full[i] + per_bond, W_MAX)

        # crypto cap
        crypto_sel = [i for i in self.crypto_idxs if i in selected]
        if crypto_sel:
            cw = w_full[crypto_sel].sum()
            if cw > CRYPTO_CAP:
                scale = CRYPTO_CAP / cw
                w_full[crypto_sel] *= scale

        # sector cap
        for sec, idxs in self.sector_idxs.items():
            if sec in ["Bond", "ETF", "Crypto"]:
                continue
            sec_sel = [i for i in idxs if i in selected]
            if sec_sel:
                sw = w_full[sec_sel].sum()
                if sw > SECTOR_CAP:
                    scale = SECTOR_CAP / sw
                    w_full[sec_sel] *= scale

        # re-normalise
        total = w_full[list(selected)].sum()
        if total > 1e-8:
            w_full[list(selected)] /= total

        return w_full


class PortfolioProblem(BasePortfolioProblem):
    """
    CONSTRAINT HANDLING 1: REPAIR
    Evolves binary z vectors. After SLSQP, calls constraints.repair()
    to project back into the feasible region.
    """
    def __init__(self, mu, scenarios, tickers):
        super().__init__(mu, scenarios, tickers, vtype=int)

    def _evaluate(self, z, out, *args, **kwargs):
        z        = np.array(z).astype(int)
        selected = np.where(z == 1)[0]

        if len(selected) == 0:
            out["F"] = [1.0, 1.0]
            return

        try:
            w_sub = self._solve_weights(selected)
        except Exception:
            out["F"] = [1.0, 1.0]
            return

        w_full            = np.zeros(self.M)
        w_full[selected]  = w_sub

        # repair projects into feasible region
        z_rep, w_rep = repair(z, w_full, self.tickers)

        out["F"] = [
            -float(np.dot(self.mu, w_rep)),
             float(cvar(w_rep, self.scenarios))
        ]


class PortfolioProblemPenalty(BasePortfolioProblem):
    """
    CONSTRAINT HANDLING 2: PENALTY
    Evolves binary z vectors. Violations add weighted penalties to objectives.
    """
    def __init__(self, mu, scenarios, tickers):
        super().__init__(mu, scenarios, tickers, vtype=int)

    def _evaluate(self, z, out, *args, **kwargs):
        z        = np.array(z).astype(int)
        selected = np.where(z == 1)[0]
        penalty  = 0.0

        # cardinality penalty
        k_diff = abs(len(selected) - K)
        if k_diff > 0:
            penalty += k_diff * 0.5

        if len(selected) == 0:
            out["F"] = [1.0 + penalty, 1.0 + penalty]
            return

        try:
            w_sub = self._solve_weights(selected)
        except Exception:
            out["F"] = [1.0 + penalty, 1.0 + penalty]
            return

        w_full           = np.zeros(self.M)
        w_full[selected] = w_sub

        raw_ret  = -float(np.dot(self.mu, w_full))
        raw_cvar =  float(cvar(w_full, self.scenarios))

        # structural penalties
        if self.bond_idxs:
            bw = w_full[self.bond_idxs].sum()
            if bw < BOND_FLOOR:
                penalty += (BOND_FLOOR - bw) * 50.0

        if self.crypto_idxs:
            cw = w_full[self.crypto_idxs].sum()
            if cw > CRYPTO_CAP:
                penalty += (cw - CRYPTO_CAP) * 50.0

        for sec, idxs in self.sector_idxs.items():
            if sec in ["Bond", "ETF", "Crypto"]:
                continue
            sw = w_full[idxs].sum()
            if sw > SECTOR_CAP:
                penalty += (sw - SECTOR_CAP) * 50.0

        out["F"] = [raw_ret + penalty, raw_cvar + penalty]


class PortfolioProblemDecoder(BasePortfolioProblem):
    """
    CONSTRAINT HANDLING 3: DECODER (Preserving Strategy)
    Evolves continuous priority scores x in [0,1]^M.
    Top-K assets by score are selected — cardinality guaranteed by construction.
    Bond/crypto/sector constraints enforced post-SLSQP.
    """
    def __init__(self, mu, scenarios, tickers):
        super().__init__(mu, scenarios, tickers, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        # decode: top-K indices by priority score
        selected = np.argsort(x)[-K:]

        # Include a bond in the selected subset when the bond floor is active,
        # so the downstream SLSQP weight optimisation can satisfy the floor.
        has_bond = any(i in self.bond_idxs for i in selected)
        if not has_bond and self.bond_idxs:
            # replace the lowest-priority selected asset with the
            # highest-priority bond
            bond_priorities = [(i, x[i]) for i in self.bond_idxs]
            best_bond       = max(bond_priorities, key=lambda t: t[1])[0]
            lowest_sel      = selected[np.argmin(x[selected])]
            selected        = np.array(
                [i for i in selected if i != lowest_sel] + [best_bond]
            )

        try:
            w_sub = self._solve_weights(selected)
        except Exception:
            out["F"] = [1.0, 1.0]
            return

        # map back to full weight vector
        w_full           = np.zeros(self.M)
        w_full[selected] = w_sub

        # enforce bond/crypto/sector constraints post-SLSQP
        w_full = self._apply_portfolio_constraints(w_full, set(selected))

        mu_sub  = self.mu[selected]
        sc_sub  = self.scenarios[:, selected]
        w_sub_f = w_full[selected]

        # ── guard: re-normalise if constraints shifted the sum ─────────
        if w_sub_f.sum() > 1e-8:
            w_sub_f = w_sub_f / w_sub_f.sum()

        out["F"] = [
            -float(np.dot(mu_sub, w_sub_f)),
             float(cvar(w_sub_f, sc_sub))
        ]
