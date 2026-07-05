"""
Return preprocessing utilities.

`main_pipeline.py` starts from adjusted close prices, then uses this
module to create the three standard inputs used downstream:

* daily log returns for scenario-based CVaR,
* annualised mean returns for expected-return objectives,
* annualised covariance for analysis or future extensions.
"""

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert adjusted close prices into daily log returns.

    Args:
        prices: Price panel indexed by date with one column per asset.

    Returns:
        Daily log returns with shape (T - 1, M), after dropping the first
        row and any rows with missing values introduced by shifting.
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualised_mean(ret: pd.DataFrame) -> pd.Series:
    """Return annualised mean log return per asset."""
    return ret.mean() * TRADING_DAYS


def covariance_matrix(ret: pd.DataFrame) -> pd.DataFrame:
    """Return the annualised sample covariance matrix of asset returns."""
    return ret.cov() * TRADING_DAYS


def compute_all(prices: pd.DataFrame):
    """
    Build all return statistics required by the optimisation pipeline.

    Returns:
        Tuple `(ret, mu, Sigma)` where `ret` is the daily return panel,
        `mu` is annualised expected return, and `Sigma` is annualised
        covariance.
    """
    ret = log_returns(prices)
    mu = annualised_mean(ret)
    Sigma = covariance_matrix(ret)

    return ret, mu, Sigma
