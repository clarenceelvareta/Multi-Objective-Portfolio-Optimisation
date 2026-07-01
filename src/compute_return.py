import numpy as np
import pandas as pd

TRADING_DAYS = 252  # annualisation factor

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. Shape: (T-1, M)."""
    return np.log(prices / prices.shift(1)).dropna()

def annualised_mean(ret: pd.DataFrame) -> pd.Series:
    """µ: annualised expected return per asset. Shape: (M,)."""
    return ret.mean() * TRADING_DAYS

def covariance_matrix(ret: pd.DataFrame) -> pd.DataFrame:
    """Σ: annualised covariance matrix. Shape: (M, M)."""
    return ret.cov() * TRADING_DAYS

def compute_all(prices: pd.DataFrame):
    """
    Convenience function — returns everything downstream needs.
    Returns: ret (T-1, M), mu (M,), Sigma (M, M)
    """
    ret   = log_returns(prices)
    mu    = annualised_mean(ret)
    Sigma = covariance_matrix(ret)
    
    return ret, mu, Sigma