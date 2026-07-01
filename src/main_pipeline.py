# main_pipeline.py
import numpy as np
from fetch_universe  import fetch_prices
from compute_return import compute_all
from compute_cvar    import cvar, expected_return
from constraints     import is_feasible, repair
from stress_window  import tariff_window, covid_window, stress_metrics
from config import ALL_TICKERS, K

def main():
    # 1. Prices
    prices = fetch_prices()
    
    # --- SAFETY DROP: Remove any tickers that yfinance failed to download ---
    prices = prices.dropna(axis=1, how='all')
    
    tickers = list(prices.columns)
    M = len(tickers)
    print(f"\n✓ Universe: {M} assets, {len(prices)} trading days")

    # 2. Returns
    ret, mu, Sigma = compute_all(prices)
    scenarios = ret.values
    
    # --- IRONCLAD CHECK: Force shapes to match exactly ---
    assert scenarios.shape[1] == M, (
        f"SHAPE MISMATCH! Returns matrix has {scenarios.shape[1]} columns, "
        f"but prices has {M} columns. Check compute_all() for hidden extra columns."
    )

    # 3. CVaR on equal-weight
    w_eq = np.ones(M) / M
    print(f"✓ Equal-weight CVaR95 = {cvar(w_eq, scenarios):.4f}")
    print(f"✓ Equal-weight E[r]   = {expected_return(w_eq, mu):.4f}")

    # 4. Constraint check on equal-weight AND a random test
    # print(f"DEBUG: w_eq sum is exactly {w_eq.sum():.4f} (Should be 1.0000)")
    
    # Test the random portfolio
    rng = np.random.default_rng(42)
    z = np.zeros(M, dtype=int)
    z[rng.choice(M, K, replace=False)] = 1
    w_rand = np.zeros(M)
    w_rand[z == 1] = 1.0 / K
    z, w_rand = repair(z, w_rand, tickers)
    
    # print(f"DEBUG: w_rand sum is {w_rand.sum():.4f} (If > 1.0, the bug is inside repair())")
    
    feasible, reasons = is_feasible(z, w_rand, tickers)
    print(f"✓ Repair test feasible: {feasible}")
    if not feasible:
        for r in reasons:
            print(f"  ✗ {r}")

    # 5. Stress windows
    # COVID window uses the main (crypto-included) universe/weights as before.
    ret_covid = covid_window(ret)
    covid_m   = stress_metrics(w_eq, ret_covid, mu)
    print(f"✓ COVID stress: CVaR={covid_m['realised_cvar']:.4f}, "
          f"MaxDD={covid_m['max_drawdown']:.4f}, "
          f"Sharpe={covid_m['sharpe_ratio']:.4f}")
    
    
    # Trump Tariff window
    ret_tariff = tariff_window(ret)
    tariff_m   = stress_metrics(w_eq, ret_tariff, mu)
    if tariff_m["realised_cvar"] is None:
        print(f"✗ Tariff stress: {tariff_m['note']}")
    else:
        print(f"✓ Tariff stress: CVaR={tariff_m['realised_cvar']:.4f}, "
              f"MaxDD={tariff_m['max_drawdown']:.4f}, "
              f"Sharpe={tariff_m['sharpe_ratio']:.4f}")

if __name__ == "__main__":
    main()