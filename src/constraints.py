# src/constraints.py
import numpy as np
from config import (
    K, W_MIN, W_MAX,
    SECTOR_CAP, CRYPTO_CAP, BOND_FLOOR,
    SECTOR_MAP, BOND_ETFS, CRYPTOS
)

def get_sector_indices(tickers: list) -> dict:
    """Returns {sector_name: [indices]} for the given ticker list."""
    idx_map = {}
    for i, t in enumerate(tickers):
        sector = SECTOR_MAP.get(t, "Unknown")
        idx_map.setdefault(sector, []).append(i)
    return idx_map

def is_feasible(z: np.ndarray, w: np.ndarray,
                tickers: list) -> tuple[bool, list]:
    """
    Returns (True, []) if feasible, or (False, [reasons]) if not.
    Useful for debugging.
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
    Project a (z, w) pair into the feasible region.
    Used by NSGA-II and MOEA/D after crossover/mutation.

    Steps:
    1. Enforce cardinality — keep top-K by weight if z.sum() > K
    2. Zero out weights for non-selected assets
    3. Normalise to sum = 1  <-- MOVED HERE
    4. Clip weights to [W_MIN, W_MAX]
    5. Enforce bond floor — if bond weight < BOND_FLOOR, top up
    6. Enforce crypto cap — if crypto weight > CRYPTO_CAP, scale down
    7. Final Normalise to sum = 1 <-- ADDED AT THE END
    """
    z = z.copy().astype(int)
    w = w.copy()

    # Step 1: cardinality
    if z.sum() > K:
        selected = np.where(z == 1)[0]
        # drop lowest-weight assets first
        drop = selected[np.argsort(w[selected])][:int(z.sum()) - K]
        z[drop] = 0

    # Step 2: zero out non-selected
    w[z == 0] = 0.0

    # If nothing selected, pick the asset with the highest weight
    if z.sum() == 0:
        best = np.argmax(w)
        z[best] = 1

    selected = np.where(z == 1)[0]

    # Step 3: Initial normalise (so floors/caps are calculated on a 100% portfolio)
    total = w[selected].sum()
    if total > 0:
        w[selected] /= total

    # Step 4: clip to [W_MIN, W_MAX]
    w[selected] = np.clip(w[selected], W_MIN, W_MAX)

    # Step 5: bond floor
    bond_idxs = [i for i, t in enumerate(tickers) if t in BOND_ETFS]
    bond_selected = [i for i in bond_idxs if z[i] == 1]

    # If no bond is currently selected, force one in by swapping out the
    # current lowest-weight holding.
    if not bond_selected and bond_idxs and len(selected) > 0:
        lowest = selected[np.argsort(w[selected])[0]]
        
        # FIX: Actually zero out the weight of the asset we are dropping!
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

    # Step 6: crypto cap
    crypto_idxs = [i for i, t in enumerate(tickers) if t in CRYPTOS]
    crypto_selected = [i for i in crypto_idxs if z[i] == 1]
    crypto_weight = w[crypto_selected].sum() if crypto_selected else 0.0
    if crypto_weight > CRYPTO_CAP and crypto_selected:
        scale = CRYPTO_CAP / crypto_weight
        w[crypto_selected] *= scale

    # Step 7: Final normalise (because adding deficits or scaling crypto breaks the sum=1)
    selected = np.where(z == 1)[0]
    total = w[selected].sum()
    if total > 0:
        w[selected] /= total

    return z, w