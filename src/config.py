"""
Central configuration for the HST Project 1 portfolio study.

This file defines the asset universe, date windows, portfolio
constraints, and sector mappings used across the pipeline. Keeping these
values in one module makes the optimisation experiments reproducible and
prevents different scripts from silently using different assumptions.
"""

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
SP500_EQUITIES = ['NVDA', 'AAPL', 'AMD', 'INTC', 'MSFT', 'AVGO', 'MU', 'CSCO', 'LRCX', 'SMCI', 'KLAC', 'HPE', 'PFE', 'BMY', 'MRK', 'GILD', 'BSX', 'JNJ', 'VTRS', 'CVS', 'ABBV', 'ABT', 'BAC', 'WFC', 'C', 'JPM', 'HBAN', 'RF', 'KEY', 'PYPL', 'XYZ', 'MS', 'TSLA', 'AMZN', 'F', 'CMG', 'CCL', 'GM', 'NCLH', 'BKNG', 'CSX', 'GE', 'DAL', 'UAL', 'FAST', 'BA', 'LUV', 'RTX', 'NFLX', 'T', 'GOOGL', 'GOOG', 'META', 'CMCSA', 'VZ', 'WMT', 'KO', 'MO', 'KR', 'PG', 'MDLZ', 'KHC', 'XOM', 'KMI', 'OXY', 'HAL', 'SLB', 'CVX', 'PCG', 'NEE', 'EXC', 'AES', 'PPL', 'HST', 'KIM', 'WY', 'DOC', 'FCX', 'NEM', 'MOS']  # from data.ipynb output

ALL_TICKERS = SP500_EQUITIES + BOND_ETFS + BROAD_ETFS + CRYPTOS


# Sector map: ticker -> GICS sector string
# Fill this in after equity screening
SECTOR_MAP = {
    # --- Information Technology ---
    "NVDA": "Information Technology",
    "AAPL": "Information Technology",
    "AMD": "Information Technology",
    "INTC": "Information Technology",
    "MSFT": "Information Technology",
    "AVGO": "Information Technology",
    "MU": "Information Technology",
    "CSCO": "Information Technology",
    "LRCX": "Information Technology",
    "SMCI": "Information Technology",
    "KLAC": "Information Technology",
    "HPE": "Information Technology",
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "CMCSA": "Communication Services",
    
    # --- Health Care ---
    "PFE": "Health Care",
    "BMY": "Health Care",
    "MRK": "Health Care",
    "GILD": "Health Care",
    "BSX": "Health Care",
    "JNJ": "Health Care",
    "VTRS": "Health Care",
    "CVS": "Health Care",
    "ABBV": "Health Care",
    "ABT": "Health Care",
    
    # --- Financials ---
    "BAC": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "JPM": "Financials",
    "HBAN": "Financials",
    "RF": "Financials",
    "KEY": "Financials",
    "PYPL": "Financials",
    "MS": "Financials",
    
    # --- Consumer Discretionary ---
    "TSLA": "Consumer Discretionary",
    "AMZN": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    "CMG": "Consumer Discretionary",
    "CCL": "Consumer Discretionary",
    "GM": "Consumer Discretionary",
    "NCLH": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    
    # --- Industrials ---
    "CSX": "Industrials",
    "GE": "Industrials",
    "DAL": "Industrials",
    "UAL": "Industrials",
    "FAST": "Industrials",
    "BA": "Industrials",
    "LUV": "Industrials",
    "RTX": "Industrials",
    
    # --- Communication Services ---
    "T": "Communication Services",
    "VZ": "Communication Services",
    
    # --- Consumer Staples ---
    "WMT": "Consumer Staples",
    "KO": "Consumer Staples",
    "MO": "Consumer Staples",
    "KR": "Consumer Staples",
    "PG": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "KHC": "Consumer Staples",
    
    # --- Energy ---
    "XOM": "Energy",
    "KMI": "Energy",
    "OXY": "Energy",
    "HAL": "Energy",
    "SLB": "Energy",
    "CVX": "Energy",
    
    # --- Utilities ---
    "PCG": "Utilities",
    "NEE": "Utilities",
    "EXC": "Utilities",
    "AES": "Utilities",
    "PPL": "Utilities",
    
    # --- Real Estate ---
    "HST": "Real Estate",
    "KIM": "Real Estate",
    "WY": "Real Estate",
    "DOC": "Real Estate",
    
    # --- Materials ---
    "FCX": "Materials",
    "DD": "Materials",
    "NEM": "Materials",
    
    # --- Asset Classes (Keep these at the bottom) ---
    **{t: "Bond" for t in BOND_ETFS},
    **{t: "ETF"  for t in BROAD_ETFS},
    **{t: "Crypto" for t in CRYPTOS},
}


if __name__ == "__main__":
    print(f"Total tickers in config: {ALL_TICKERS}")
