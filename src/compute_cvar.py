import numpy as np
import pandas as pd
from config import CVAR_ALPHA

def portfolio_returns(weights: np.ndarray,
                      scenarios: np.ndarray) -> np.ndarray:
    """
    weights:   shape (M,) — portfolio weights
    scenarios: shape (T, M) — daily log returns as numpy array
    Returns:   shape (T,) — portfolio daily returns
    """
    return scenarios @ weights  # fast matrix multiply

def cvar(weights: np.ndarray,
         scenarios: np.ndarray,
         alpha: float = CVAR_ALPHA) -> float:
    """
    CVaR_alpha: expected loss in the worst (1-alpha) fraction of days.
    Returns a POSITIVE number (loss convention).

    Steps:
    1. Compute portfolio return for each day
    2. Find the alpha-quantile (VaR threshold)
    3. Average the returns BELOW that threshold
    4. Negate to get a positive loss number
    """
    port_ret      = portfolio_returns(weights, scenarios)
    var_threshold = np.quantile(port_ret, 1 - alpha)
    tail_returns  = port_ret[port_ret <= var_threshold]
    return -tail_returns.mean()

def expected_return(weights: np.ndarray,
                    mu: np.ndarray) -> float:
    """Annualised expected return."""
    return float(weights @ mu)