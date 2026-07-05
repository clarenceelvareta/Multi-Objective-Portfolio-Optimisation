"""
Price-data loading and caching.

The main pipeline calls `fetch_prices()` to obtain adjusted close prices
for the configured universe. The function first tries the local CSV cache
under `src/raw/`; if the cache is missing or disabled, it downloads from
yfinance and writes the cache for later runs.

Running this file directly performs a small data-loading smoke check.
That mode may download market data, so use it only when you intentionally
want to refresh or inspect the price panel.
"""

import os
import yfinance as yf
import pandas as pd
from config import ALL_TICKERS, START_DATE, END_DATE


RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_prices(tickers=ALL_TICKERS,
                 start=START_DATE,
                 end=END_DATE,
                 cache=True,
                 cache_name="prices.csv") -> pd.DataFrame:
    """
    Load adjusted close prices for a ticker universe.

    Args:
        tickers: Sequence of ticker symbols accepted by yfinance.
        start: Inclusive start date passed to yfinance.
        end: Exclusive end date passed to yfinance.
        cache: When true, load from and save to `src/raw/cache_name`.
        cache_name: CSV filename for the cached price panel.

    Returns:
        DataFrame indexed by date with one adjusted-close column per
        asset. Columns with no data are dropped; small gaps are
        forward-filled and remaining missing rows are removed.

    Notes:
        The final date range can be shorter than `start` to `end` if a
        late-starting asset is included, because the pipeline requires a
        complete rectangular panel for scenario calculations.
    """
    cache_path = os.path.join(RAW_DIR, cache_name)

    if cache and os.path.exists(cache_path):
        print("Loading prices from cache...")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded {prices.shape[1]} assets, {prices.shape[0]} days from cache")
        return prices

    print(f"Downloading {len(tickers)} assets from yfinance...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    else:
        raw = raw[["Close"]]

    missing = raw.columns[raw.isnull().all()].tolist()
    if missing:
        print(f"\nWARNING: these tickers had no data and were dropped: {missing}")
    prices = raw.dropna(axis=1, how="all")

    first_valid = prices.apply(lambda col: col.first_valid_index())
    latest_start = first_valid.max()
    if latest_start is not None and latest_start > pd.Timestamp(start):
        late_starters = first_valid[first_valid > pd.Timestamp(start) + pd.Timedelta(days=30)]
        print(f"\nNOTE: panel start will be truncated to {latest_start.date()} "
              f"because of late-starting tickers:")
        print(late_starters.sort_values().to_string())

    prices = prices.ffill().dropna()

    if cache:
        prices.to_csv(cache_path)
        print(f"Prices cached to {cache_path}")

    print(f"\nFinal universe: {prices.shape[1]} assets, {prices.shape[0]} days")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices


if __name__ == "__main__":
    prices = fetch_prices()

    print("\nSample prices (last 3 days):")
    print(prices.tail(3))

    print("\nAssets loaded:")
    print(prices.columns.tolist())

    nan_cols = prices.columns[prices.isnull().any()].tolist()
    if nan_cols:
        print(f"\nWARNING: NaNs still present in: {nan_cols}")
    else:
        print("\nNo NaNs in price data")
