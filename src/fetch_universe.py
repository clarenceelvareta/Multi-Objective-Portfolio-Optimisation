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
    Returns a DataFrame of adjusted close prices.
    Shape: (T, M) — T trading days, M assets.
    Columns: ticker symbols.

    cache_name lets callers use a separate cache file so the main
    100-asset (crypto-included) universe and the crypto-free GFC-only
    universe don't overwrite each other.
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

    # fix MultiIndex columns — keep only Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    else:
        raw = raw[["Close"]]

    # drop tickers with no data at all
    missing = raw.columns[raw.isnull().all()].tolist()
    if missing:
        print(f"\nWARNING: these tickers had no data and were dropped: {missing}")
    prices = raw.dropna(axis=1, how="all")

    # WARNING: dropna(how="any") below keeps only rows where EVERY
    # remaining ticker has a price. If any ticker's first trade date is
    # late (e.g. a crypto or recent IPO), it truncates ALL other assets'
    # history down to that date too. Report the culprits before dropping.
    first_valid = prices.apply(lambda col: col.first_valid_index())
    latest_start = first_valid.max()
    if latest_start is not None and latest_start > pd.Timestamp(start):
        late_starters = first_valid[first_valid > pd.Timestamp(start) + pd.Timedelta(days=30)]
        print(f"\nNOTE: panel start will be truncated to {latest_start.date()} "
              f"because of late-starting tickers:")
        print(late_starters.sort_values().to_string())

    # forward-fill small gaps then drop any remaining NaNs
    prices = prices.ffill().dropna()

    # save to cache
    if cache:
        prices.to_csv(cache_path)
        print(f"Prices cached to {cache_path}")

    print(f"\n✓ Final universe: {prices.shape[1]} assets, {prices.shape[0]} days")
    print(f"✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices

if __name__ == "__main__":
    prices = fetch_prices()

    print("\nSample prices (last 3 days):")
    print(prices.tail(3))

    print("\nAssets loaded:")
    print(prices.columns.tolist())

    # check for any remaining NaNs
    nan_cols = prices.columns[prices.isnull().any()].tolist()
    if nan_cols:
        print(f"\nWARNING: NaNs still present in: {nan_cols}")
    else:
        print("\n✓ No NaNs in price data")