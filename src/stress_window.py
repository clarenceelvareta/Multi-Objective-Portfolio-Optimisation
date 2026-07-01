# src/stress_windows.py
import pandas as pd
from config import COVID_START, COVID_END, TARIFF_END, TARIFF_START

def tariff_window(ret_full):
    """Extract the 2025 Trump Tariff shock window returns."""
    mask = (ret_full.index >= TARIFF_START) & (ret_full.index <= TARIFF_END)
    ret_tariff = ret_full.loc[mask]
    
    if ret_tariff.empty:
        return ret_full.iloc[0:0]  # Return empty if no data
    
    return ret_tariff

def covid_window(ret: pd.DataFrame) -> pd.DataFrame:
    """COVID Feb-Apr 2020 out-of-sample returns."""
    return ret.loc[COVID_START:COVID_END]

def stress_metrics(weights, ret_window: pd.DataFrame,
                   mu: pd.Series) -> dict:
    """
    Given a weight vector and a stress-period returns DataFrame,
    compute realised CVaR, max drawdown, and Sharpe ratio.
    Returns a None-filled dict with a note if the window has no data,
    instead of crashing on an empty-array quantile.
    """
    import numpy as np
    from compute_cvar import cvar, expected_return

    if ret_window.empty:
        return {
            "realised_cvar": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
            "note": "no trading data in this window for the given universe",
        }

    scenarios = ret_window.values
    w = weights

    realised_cvar = cvar(w, scenarios)

    port_ret = scenarios @ w
    cumulative = pd.Series(1 + port_ret).cumprod()
    rolling_max = pd.Series(cumulative).cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    return {
        "realised_cvar": realised_cvar,
        "max_drawdown":  max_dd,
        "sharpe_ratio":  sharpe,
    }