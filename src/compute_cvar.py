"""
Risk and return helpers for portfolio objective evaluation.

The pipeline represents scenarios as daily log-return rows with one
column per asset. This module converts a weight vector into portfolio
scenario returns and computes CVaR using a positive-loss convention:
larger CVaR means larger downside tail loss. The expected-return helper
assumes the input `mu` is already annualised by `compute_return.py`.
"""

import numpy as np
import pandas as pd
from config import CVAR_ALPHA


def portfolio_returns(weights: np.ndarray,
                      scenarios: np.ndarray) -> np.ndarray:
    """
    Compute daily portfolio returns for all historical/scenario rows.

    Args:
        weights: Portfolio weights with shape (M,).
        scenarios: Daily log-return matrix with shape (T, M).

    Returns:
        A length-T vector of daily portfolio returns.
    """
    return scenarios @ weights


def cvar(weights: np.ndarray,
         scenarios: np.ndarray,
         alpha: float = CVAR_ALPHA) -> float:
    """
    Compute historical CVaR_alpha under a positive-loss convention.

    The lower tail is selected using the empirical (1-alpha) quantile of
    daily portfolio returns. Tail returns are averaged and negated so the
    returned value is positive when the tail has losses.
    """
    port_ret = portfolio_returns(weights, scenarios)
    var_threshold = np.quantile(port_ret, 1 - alpha)
    tail_returns = port_ret[port_ret <= var_threshold]
    return -tail_returns.mean()


def expected_return(weights: np.ndarray,
                    mu: np.ndarray) -> float:
    """Return the annualised portfolio expected return `weights @ mu`."""
    return float(weights @ mu)
