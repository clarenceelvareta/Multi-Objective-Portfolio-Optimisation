"""
Stress-window extraction and realised portfolio stress metrics.

The main study compares the equal-weight and optimised portfolios during
two out-of-sample stress periods configured in `config.py`: COVID-19 and
the April 2025 tariff shock window. This module isolates those date
windows and computes realised CVaR, maximum drawdown, and Sharpe ratio
from actual window returns.
"""

import pandas as pd
from config import COVID_START, COVID_END, TARIFF_END, TARIFF_START


def tariff_window(ret_full: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the configured April 2025 tariff shock return window.

    Returns an empty DataFrame with matching columns if the data panel
    does not cover the configured dates.
    """
    mask = (ret_full.index >= TARIFF_START) & (ret_full.index <= TARIFF_END)
    ret_tariff = ret_full.loc[mask]

    if ret_tariff.empty:
        return ret_full.iloc[0:0]

    return ret_tariff


def covid_window(ret: pd.DataFrame) -> pd.DataFrame:
    """Extract the configured COVID-19 stress return window."""
    return ret.loc[COVID_START:COVID_END]


def stress_metrics(weights, ret_window: pd.DataFrame,
                   mu: pd.Series) -> dict:
    """
    Compute realised stress-period risk and performance metrics.

    Args:
        weights: Full portfolio weight vector aligned with `ret_window`.
        ret_window: Daily returns in the stress period.
        mu: Annualised expected returns. Present for API compatibility;
            realised metrics are computed from `ret_window`.

    Returns:
        Dictionary with realised CVaR, max drawdown, and Sharpe ratio.
        If the window is empty, metric values are `None` and a note is
        included instead of raising on an empty quantile.
    """
    from compute_cvar import cvar

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
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
    }
