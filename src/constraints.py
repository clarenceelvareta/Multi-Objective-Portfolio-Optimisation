"""
Portfolio feasibility checks and repair projection.

The metaheuristic encodings can generate portfolios that violate the
project constraints. This module provides:

* `is_feasible()` for diagnostics and sanity checks,
* `repair()` for projecting candidate `(z, w)` solutions back toward a
  feasible portfolio before objective evaluation.

`z` is the binary selected-asset vector and `w` is the full weight
vector aligned with the same ticker order.
"""

import numpy as np
from config import (
    K, W_MIN, W_MAX,
    SECTOR_CAP, CRYPTO_CAP, BOND_FLOOR,
    SECTOR_MAP, BOND_ETFS, CRYPTOS
)


def get_sector_indices(tickers: list) -> dict:
    """Return a mapping `{sector_name: [asset_indices]}` for `tickers`."""
    idx_map = {}
    for i, t in enumerate(tickers):
        sector = SECTOR_MAP.get(t, "Unknown")
        idx_map.setdefault(sector, []).append(i)
    return idx_map


def is_feasible(z: np.ndarray, w: np.ndarray,
                tickers: list) -> tuple[bool, list]:
    """
    Check whether a full portfolio satisfies the configured constraints.

    Returns:
        `(True, [])` when feasible. Otherwise returns `False` and a list
        of human-readable violation messages, useful for debugging repair
        operators and selected Gurobi portfolios.
    """
    reasons = []

    if abs(w.sum() - 1.0) > 1e-6:
        reasons.append(f"Budget violated: sum(w)={w.sum():.4f}")

    if z.sum() > K:
        reasons.append(f"Cardinality violated: {int(z.sum())} > {K}")

    for i, (zi, wi) in enumerate(zip(z, w)):
        if zi == 1 and not (W_MIN <= wi <= W_MAX):
            reasons.append(f"Linkage violated at {tickers[i]}: w={wi:.4f}")
        if zi == 0 and wi != 0:
            reasons.append(f"Linkage violated: {tickers[i]} selected z=0 but w={wi:.4f}")

    sector_idx = get_sector_indices(tickers)
    for sector, idxs in sector_idx.items():
        s_weight = w[idxs].sum()
        if s_weight > SECTOR_CAP + 1e-6:
            reasons.append(f"Sector cap violated: {sector} = {s_weight:.4f}")

    crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]
    if w[crypto_idxs].sum() > CRYPTO_CAP + 1e-6:
        reasons.append(f"Crypto cap violated: {w[crypto_idxs].sum():.4f}")

    bond_idxs = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
    if w[bond_idxs].sum() < BOND_FLOOR - 1e-6:
        reasons.append(f"Bond floor violated: {w[bond_idxs].sum():.4f}")

    return len(reasons) == 0, reasons


def repair(z: np.ndarray, w: np.ndarray,
           tickers: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a candidate `(z, w)` pair back toward feasibility.

    The projection enforces cardinality, zeros non-selected assets,
    normalises weights, clips asset weights, forces at least one bond
    when bond assets exist, applies bond and crypto limits, and then
    normalises the selected weights again.

    Returns:
        Repaired `(z, w)` arrays. The function is deterministic for a
        given input and does not mutate the caller's arrays.
    """
    z = z.copy().astype(int)
    w = w.copy()

    if z.sum() > K:
        selected = np.where(z == 1)[0]
        drop = selected[np.argsort(w[selected])][:int(z.sum()) - K]
        z[drop] = 0

    w[z == 0] = 0.0

    if z.sum() == 0:
        best = np.argmax(w)
        z[best] = 1

    selected = np.where(z == 1)[0]

    total = w[selected].sum()
    if total > 0:
        w[selected] /= total

    w[selected] = np.clip(w[selected], W_MIN, W_MAX)

    bond_idxs = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
    bond_selected = [i for i in bond_idxs if z[i] == 1]

    if not bond_selected and bond_idxs and len(selected) > 0:
        lowest = selected[np.argsort(w[selected])[0]]
        w[lowest] = 0.0

        z[lowest] = 0
        chosen_bond = bond_idxs[0]
        z[chosen_bond] = 1
        w[chosen_bond] = W_MIN
        selected = np.where(z == 1)[0]
        bond_selected = [chosen_bond]

    bond_weight = w[bond_selected].sum() if bond_selected else 0.0

    if bond_weight < BOND_FLOOR and bond_selected:
        deficit = BOND_FLOOR - bond_weight
        per_bond = deficit / len(bond_selected)
        for i in bond_selected:
            w[i] = min(w[i] + per_bond, W_MAX)

    crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]
    crypto_selected = [i for i in crypto_idxs if z[i] == 1]
    crypto_weight = w[crypto_selected].sum() if crypto_selected else 0.0
    if crypto_weight > CRYPTO_CAP and crypto_selected:
        scale = CRYPTO_CAP / crypto_weight
        w[crypto_selected] *= scale

    selected = np.where(z == 1)[0]
    total = w[selected].sum()
    if total > 0:
        w[selected] /= total

    return z, w
