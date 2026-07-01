# config.py

# Universe
M = 100
K = 15

# Date range
START_DATE = "2015-01-01"
END_DATE   = "2025-05-20"  # Changed to capture April 2025 tariff data

# Stress windows
TARIFF_START = "2025-04-02"  # "Liberation Day" tariff announcements began
TARIFF_END   = "2025-04-30"  # Capture the immediate market shock/reaction
COVID_START  = "2020-02-01"
COVID_END    = "2020-04-30"

# Constraints
W_MIN       = 0.02   # linkage lower bound
W_MAX       = 0.30   # linkage upper bound
SECTOR_CAP  = 0.40
CRYPTO_CAP  = 0.20
BOND_FLOOR  = 0.10
CVAR_ALPHA  = 0.95   # CVaR at 95th percentile

# Asset tickers — exactly 100
BOND_ETFS = ["AGG", "TLT", "BND", "LQD", "HYG", "SHY"]

BROAD_ETFS = [
    "SPY", "QQQ", "IWM", "GLD", "VNQ",
    "EEM", "DBC", "TIP", "PDBC", "SCHD"
]

CRYPTOS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]

# You'll fill in the 80 equities after running the screening step
# Placeholder — replace with your final screened 80 tickers
SP500_EQUITIES = ['NVDA', 'AAPL', 'AMD', 'INTC', 'MSFT', 'AVGO', 'MU', 'CSCO', 'LRCX', 'SMCI', 'KLAC', 'HPE', 'PFE', 'BMY', 'MRK', 'GILD', 'BSX', 'JNJ', 'VTRS', 'CVS', 'ABBV', 'ABT', 'BAC', 'WFC', 'C', 'JPM', 'HBAN', 'RF', 'KEY', 'PYPL', 'XYZ', 'MS', 'TSLA', 'AMZN', 'F', 'CMG', 'CCL', 'GM', 'NCLH', 'BKNG', 'CSX', 'GE', 'DAL', 'UAL', 'FAST', 'BA', 'LUV', 'RTX', 'NFLX', 'T', 'GOOGL', 'GOOG', 'META', 'CMCSA', 'VZ', 'WMT', 'KO', 'MO', 'KR', 'PG', 'MDLZ', 'KHC', 'XOM', 'KMI', 'OXY', 'HAL', 'SLB', 'CVX', 'PCG', 'NEE', 'EXC', 'AES', 'PPL', 'HST', 'KIM', 'WY', 'DOC', 'FCX', 'NEM', 'MOS']  # from data.ipynb output

ALL_TICKERS = SP500_EQUITIES + BOND_ETFS + BROAD_ETFS + CRYPTOS


# Sector map: ticker -> GICS sector string
# Fill this in after equity screening
SECTOR_MAP = {
    # e.g. "AAPL": "Information Technology",
    # "JPM":  "Financials",
    # ...
    **{t: "Bond" for t in BOND_ETFS},
    **{t: "ETF"  for t in BROAD_ETFS},
    **{t: "Crypto" for t in CRYPTOS},
}

if __name__ == "__main__":
    print(f"Total tickers in config: {ALL_TICKERS}")