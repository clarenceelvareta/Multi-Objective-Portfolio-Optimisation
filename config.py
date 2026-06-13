# config.py

# Universe
M = 100
K = 15

# Date range
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# Stress windows
GFC_START   = "2008-09-01"
GFC_END     = "2009-03-31"
COVID_START = "2020-02-01"
COVID_END   = "2020-04-30"

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
SP500_EQUITIES = [
    # ... 80 tickers across all 11 GICS sectors
]

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