# main_pipeline.py
import numpy as np
import sys  # Added to allow sys.exit() to stop the script after Gurobi test
from fetch_universe  import fetch_prices
from compute_return import compute_all
from compute_cvar    import cvar, expected_return
from constraints     import is_feasible, repair
from stress_window  import covid_window, tariff_window, stress_metrics
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
    print(f"✓ Returns computed: mu range [{mu.min():.3f}, {mu.max():.3f}]")

    # 3. CVaR on equal-weight
    w_eq = np.ones(M) / M
    print(f"✓ Equal-weight CVaR95 = {cvar(w_eq, scenarios):.4f}")
    print(f"✓ Equal-weight E[r]   = {expected_return(w_eq, mu):.4f}")

    # 4. Constraint check on equal-weight AND a random test
    print(f"DEBUG: w_eq sum is exactly {w_eq.sum():.4f} (Should be 1.0000)")
    
    # Test the random portfolio
    rng = np.random.default_rng(42)
    z = np.zeros(M, dtype=int)
    z[rng.choice(M, K, replace=False)] = 1
    w_rand = np.zeros(M)
    w_rand[z == 1] = 1.0 / K
    z, w_rand = repair(z, w_rand, tickers)
    
    print(f"DEBUG: w_rand sum is {w_rand.sum():.4f} (If > 1.0, the bug is inside repair())")
    
    feasible, reasons = is_feasible(z, w_rand, tickers)
    print(f"✓ Baseline constraints check: {feasible} (Expected False for un-optimized random weights)")
    if not feasible:
        for r in reasons:
            print(f"  ✗ {r}")

    # 5. Stress windows
    # COVID window
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

    # =====================================================================
    # 6. GROUND TRUTH: Gurobi Exact Optimization
    # =====================================================================
    print("\n" + "="*50)
    print("Running Gurobi Exact Optimization (M=100)...")
    print("Note: This may take 1-5 minutes as it solves a Mixed-Integer program.")
    print("="*50)
    
    # Import locally so it doesn't slow down the rest of the script if not needed
    from gurobi_optimizer import optimize_gurobi 
    
    # We ask for 5 points on the Pareto front to start
    # Epsilon bounds CVaR. Equal weight CVaR is ~0.023, so we sweep around that.
    epsilons = np.linspace(0.015, 0.04, 5) 
    
    fronts, time_taken = optimize_gurobi(mu, scenarios, tickers, epsilon_values=epsilons)
    
    print(f"\n✓ Gurobi finished in {time_taken:.2f} seconds")
    print(f"✓ Found {len(fronts)} Pareto-optimal points:")
    for i, pt in enumerate(fronts):
        print(f"  Point {i+1}: E[r]={pt['ret']:.4f}, CVaR={pt['cvar']:.4f}, Assets held={pt['z'].sum()}")

    # Stop the script here for now so you don't have to wait for Gurobi 
    # every time you run the pipeline while building NSGA-II next.
    # print("\n[Pipeline paused. Remove sys.exit() in main_pipeline.py to continue.]")
    # sys.exit()

    # =====================================================================
    # 7. METAHEURISTIC: NSGA-II Optimization
    # =====================================================================
    print("\n" + "="*50)
    print("Running NSGA-II Metaheuristic...")
    print("="*50)
    
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize as pymoo_minimize
    from nsga2_optimizer import PortfolioProblem

    problem = PortfolioProblem(mu, scenarios, tickers)

    algorithm = NSGA2(
        pop_size=100,              # Population size (100 portfolios per generation)
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=int),
        mutation=PM(eta=20, vtype=int),
        eliminate_duplicates=True
    )

    # Run for 50 generations (100 * 50 = 5000 portfolio evaluations)
    res = pymoo_minimize(problem, algorithm, ('n_gen', 50), seed=42, verbose=False)

    print(f"✓ NSGA-II finished.")
    print(f"✓ Found {len(res.F)} non-dominated Pareto points:")
    
    # Sort the results by Return to print them nicely
    sorted_idx = np.argsort(res.F[:, 0] * -1) # Sort descending by return (remember it's negative)
    
    for i, idx in enumerate(sorted_idx[:5]): # Print top 5
        z_nsga = res.X[idx].astype(int)
        w_nsga = np.zeros(M)
        
        # Reconstruct weights for printing (simplified: assume equal weight for display)
        selected = np.where(z_nsga == 1)[0]
        w_nsga[selected] = 1.0 / len(selected)
        
        ret_val = -res.F[idx, 0]
        cvar_val = res.F[idx, 1]
        
        print(f"  Point {i+1}: E[r]={ret_val:.4f}, CVaR={cvar_val:.4f}, Assets held={len(selected)}")

    # =====================================================================
    # 8. EVALUATION METRICS (HV, GD)
    # =====================================================================
    print("\n" + "="*50)
    print("Calculating Pareto Front Metrics...")
    print("="*50)
    
    from pymoo.indicators.hv import HV
    from pymoo.indicators.gd import GD

    # 1. Format the Gurobi Front (The Reference/True Front)
    # Gurobi objectives: [Return, CVaR]
    gurobi_F = np.array([[pt['ret'], pt['cvar']] for pt in fronts])
    
    # Format NSGA-II Front
    # NSGA-II objectives are stored as [-Return, CVaR], so we fix the sign
    nsga_F = res.F.copy()
    nsga_F[:, 0] = -nsga_F[:, 0] # Convert back to positive Return

    # 2. Calculate Hypervolume (HV)
    # We need a "worst point" (reference point) that is worse than all data
    ref_point = np.array([0.6, 0.10]) # Return=0.6, CVaR=0.10
    hv_indicator = HV(ref_point=ref_point)
    
    hv_gurobi = hv_indicator.do(gurobi_F)
    hv_nsga = hv_indicator.do(nsga_F)
    
    print(f"✓ Hypervolume (Higher is better):")
    print(f"  Gurobi (Exact):  {hv_gurobi:.6f}")
    print(f"  NSGA-II (Approx): {hv_nsga:.6f}")

    # 3. Calculate Generational Distance (GD)
    # Measures how far the NSGA-II points are from the Gurobi points (Lower is better)
    gd_indicator = GD(gurobi_F)
    gd_nsga = gd_indicator.do(nsga_F)
    
    print(f"\n✓ Generational Distance to Gurobi Front (Lower is better):")
    print(f"  NSGA-II GD: {gd_nsga:.6f}")

    print("\n[End of Pipeline]")


if __name__ == "__main__":
    main()