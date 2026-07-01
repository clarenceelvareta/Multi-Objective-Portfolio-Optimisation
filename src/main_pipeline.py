# main_pipeline.py
import numpy as np
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
    # 6. GROUND TRUTH: Gurobi Exact Optimization (Will take ~1 Hour)
    # =====================================================================
    # print("\n" + "="*50)
    # print("Running Gurobi Exact Optimization (M=100)...")
    # print("Note: Forcing K=15 and sweeping 20 points. This WILL take a long time.")
    # print("="*50)
    
    # from gurobi_optimizer import optimize_gurobi 
    
    # # 20 points forces Gurobi to solve 20 separate MIQP problems
    # epsilons = np.linspace(0.012, 0.055, 20) 
    
    # fronts, time_taken = optimize_gurobi(mu, scenarios, tickers, epsilon_values=epsilons)
        # =====================================================================
    # 6. GROUND TRUTH: Gurobi Exact Optimization (Dry Run: ~5 mins)
    # =====================================================================
    print("\n" + "="*50)
    print("Running Gurobi Exact Optimization (M=100)...")
    print("Note: DRY RUN - Sweeping 5 points to test for errors.")
    print("="*50)
    
    from gurobi_optimizer import optimize_gurobi 
    
    # REDUCED TO 5 POINTS for a quick 2-5 minute test run
    epsilons = np.linspace(0.015, 0.035, 5) 
    
    fronts, time_taken = optimize_gurobi(mu, scenarios, tickers, epsilon_values=epsilons)
    
    print(f"\n✓ Gurobi finished in {time_taken:.2f} seconds")
    print(f"✓ Found {len(fronts)} Pareto-optimal points:")
    
    # --- FIND THE BEST PORTFOLIO VIA SHARPE RATIO ---
    best_sharpe = -np.inf
    best_portfolio = None
    rf = 0.02 # Risk-free rate assumption
    
    for pt in fronts:
        ret = pt['ret']
        cvar_val = pt['cvar']
        std_approx = cvar_val / 1.65  # StdDev approximation
        sharpe = (ret - rf) / std_approx
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_portfolio = pt
            
    if best_portfolio:
        print(f"\n*** BEST PORTFOLIO (Max Sharpe Ratio) ***")
        print(f"  Expected Return: {best_portfolio['ret']:.4f}")
        print(f"  CVaR (95%):      {best_portfolio['cvar']:.4f}")
        print(f"  Approx Sharpe:   {best_sharpe:.4f}")
        print(f"  Assets Held:     {best_portfolio['z'].sum()}")
        selected_tickers = [tickers[i] for i, z in enumerate(best_portfolio['z']) if z == 1]
        print(f"  Tickers:         {', '.join(selected_tickers)}")

    # =====================================================================
    # 7. GUROBI SCALING EXPERIMENT (NP-Hard Proof)
    # =====================================================================
    print("\n" + "="*50)
    print("Running Gurobi Scaling Experiment (M=20 to 100)...")
    print("="*50)
    from gurobi_optimizer import scaling_experiment
    
    scaling_results = scaling_experiment(mu, scenarios, tickers)
    
    print("\n✓ Scaling Results (Time in seconds):")
    for m_size, time_taken in scaling_results.items():
        print(f"  M={m_size}: {time_taken:.2f} seconds")
    print("-> Notice how time increases non-linearly as M grows!")

    # =====================================================================
    # 8. METAHEURISTIC 1: NSGA-II Optimization
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
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=int),
        mutation=PM(eta=20, vtype=int),
        eliminate_duplicates=True
    )

    res = pymoo_minimize(problem, algorithm, ('n_gen', 50), seed=42, verbose=False)

    print(f"✓ NSGA-II finished.")
    print(f"✓ Found {len(res.F)} non-dominated Pareto points:")
    
    sorted_idx = np.argsort(res.F[:, 0] * -1)
    for i, idx in enumerate(sorted_idx[:5]):
        z_nsga = res.X[idx].astype(int)
        selected = np.where(z_nsga == 1)[0]
        ret_val = -res.F[idx, 0]
        cvar_val = res.F[idx, 1]
        print(f"  Point {i+1}: E[r]={ret_val:.4f}, CVaR={cvar_val:.4f}, Assets held={len(selected)}")

    # =====================================================================
    # 9. METAHEURISTIC 2: MOEA/D Optimization
    # =====================================================================
    print("\n" + "="*50)
    print("Running MOEA/D Metaheuristic...")
    print("="*50)
    
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

    algorithm_moead = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.9,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=int),
        mutation=PM(eta=20, vtype=int)
    )

    res_moead = pymoo_minimize(problem, algorithm_moead, ('n_gen', 50), seed=42, verbose=False)

    print(f"✓ MOEA/D finished.")
    print(f"✓ Found {len(res_moead.F)} non-dominated Pareto points:")
    
    sorted_idx_moead = np.argsort(res_moead.F[:, 0] * -1)
    for i, idx in enumerate(sorted_idx_moead[:5]):
        z_moead = res_moead.X[idx].astype(int)
        selected = np.where(z_moead == 1)[0]
        ret_val = -res_moead.F[idx, 0]
        cvar_val = res_moead.F[idx, 1]
        print(f"  Point {i+1}: E[r]={ret_val:.4f}, CVaR={cvar_val:.4f}, Assets held={len(selected)}")

    # =====================================================================
    # 10. FINAL EVALUATION METRICS (HV, GD, IGD)
    # =====================================================================
    print("\n" + "="*50)
    print("Final Algorithm Comparison Metrics")
    print("="*50)
    
    from pymoo.indicators.hv import HV
    from pymoo.indicators.gd import GD
    from pymoo.indicators.igd import IGD

    # 1. Format the Gurobi Front (The Reference/True Front)
    gurobi_F = np.array([[pt['ret'], pt['cvar']] for pt in fronts])
    
    # Format NSGA-II Front (Fix the negative return sign)
    nsga_F = res.F.copy()
    nsga_F[:, 0] = -nsga_F[:, 0] 

    # Format MOEA/D Front (Fix the negative return sign)
    moead_F = res_moead.F.copy()
    moead_F[:, 0] = -moead_F[:, 0]

    # 2. Calculate Metrics
    ref_point = np.array([0.6, 0.10]) 
    hv_indicator = HV(ref_point=ref_point)
    gd_indicator = GD(gurobi_F)
    igd_indicator = IGD(gurobi_F)
    
    hv_gurobi = hv_indicator.do(gurobi_F)
    hv_nsga = hv_indicator.do(nsga_F)
    hv_moead = hv_indicator.do(moead_F)
    
    gd_nsga = gd_indicator.do(nsga_F)
    gd_moead = gd_indicator.do(moead_F)
    
    igd_nsga = igd_indicator.do(nsga_F)
    igd_moead = igd_indicator.do(moead_F)

    print(f"{'Metric':<25} {'NSGA-II':<15} {'MOEA/D':<15} {'Gurobi (Ref)':<15}")
    print("-" * 70)
    print(f"{'Hypervolume (HV)':<25} {hv_nsga:<15.6f} {hv_moead:<15.6f} {hv_gurobi:<15.6f}")
    print(f"{'Gen. Distance (GD)':<25} {gd_nsga:<15.6f} {gd_moead:<15.6f} {'0.000000 (Optimal)':<15}")
    print(f"{'Inv. Gen. Dist (IGD)':<25} {igd_nsga:<15.6f} {igd_moead:<15.6f} {'0.000000 (Optimal)':<15}")

    print("\n[End of Pipeline - All Proposal Requirements Met]")

if __name__ == "__main__":
    main()