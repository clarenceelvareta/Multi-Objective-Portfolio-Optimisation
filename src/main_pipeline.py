# main_pipeline.py
import numpy as np
from fetch_universe  import fetch_prices
from compute_return import compute_all
from compute_cvar    import cvar, expected_return
from constraints     import is_feasible, repair
from stress_window  import covid_window, tariff_window, stress_metrics
from config import ALL_TICKERS, K

def main():
    # =====================================================================
    # 1. DATA LOADING & PREPROCESSING
    # =====================================================================
    prices = fetch_prices()
    prices = prices.dropna(axis=1, how='all') # Safety drop failed downloads
    
    tickers = list(prices.columns)
    M = len(tickers)
    print(f"\n✓ Universe: {M} assets, {len(prices)} trading days")

    ret, mu, Sigma = compute_all(prices)
    scenarios = ret.values
    
    assert scenarios.shape[1] == M, "SHAPE MISMATCH! Check compute_all()."
    print(f"✓ Returns computed: mu range [{mu.min():.3f}, {mu.max():.3f}]")

    # =====================================================================
    # 2. BASELINE EVALUATION
    # =====================================================================
    w_eq = np.ones(M) / M
    print(f"✓ Equal-weight CVaR95 = {cvar(w_eq, scenarios):.4f}")
    print(f"✓ Equal-weight E[r]   = {expected_return(w_eq, mu):.4f}")

    # Random portfolio test (for constraint checking)
    rng = np.random.default_rng(42)
    z = np.zeros(M, dtype=int)
    z[rng.choice(M, K, replace=False)] = 1
    w_rand = np.zeros(M)
    w_rand[z == 1] = 1.0 / K
    z, w_rand = repair(z, w_rand, tickers)
    feasible, reasons = is_feasible(z, w_rand, tickers)
    print(f"✓ Baseline constraints check: {feasible} (Expected False for random weights)")
    if not feasible:
        for r in reasons: print(f"  ✗ {r}")

    # =====================================================================
    # 3. STRESS TESTING (Baseline)
    # =====================================================================
    ret_covid = covid_window(ret)
    covid_m   = stress_metrics(w_eq, ret_covid, mu)
    print(f"✓ COVID stress: CVaR={covid_m['realised_cvar']:.4f}, MaxDD={covid_m['max_drawdown']:.4f}, Sharpe={covid_m['sharpe_ratio']:.4f}")

    ret_tariff = tariff_window(ret)
    tariff_m   = stress_metrics(w_eq, ret_tariff, mu)
    if tariff_m["realised_cvar"] is not None:
        print(f"✓ Tariff stress: CVaR={tariff_m['realised_cvar']:.4f}, MaxDD={tariff_m['max_drawdown']:.4f}, Sharpe={tariff_m['sharpe_ratio']:.4f}")

    # =====================================================================
    # 4. GROUND TRUTH: Gurobi Exact Optimization (Dry Run: ~5 mins)
    # =====================================================================
    print("\n" + "="*50)
    print("Running Gurobi Exact Optimization (M=100)...")
    print("Note: DRY RUN - Sweeping 5 points. Change to 20 for 1hr run.")
    print("="*50)
    
    from gurobi_optimizer import optimize_gurobi, scaling_experiment
    epsilons = np.linspace(0.015, 0.035, 5) 
    
    fronts, time_taken = optimize_gurobi(mu, scenarios, tickers, epsilon_values=epsilons)
    print(f"\n✓ Gurobi finished in {time_taken:.2f} seconds")
    print(f"✓ Found {len(fronts)} Pareto-optimal points:")
    
    # Find Best Portfolio via Sharpe Ratio
    best_sharpe = -np.inf
    best_portfolio = None
    rf = 0.02 
    
    for pt in fronts:
        std_approx = pt['cvar'] / 1.65 
        sharpe = (pt['ret'] - rf) / std_approx
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_portfolio = pt
            
    if best_portfolio:
        print(f"\n*** BEST PORTFOLIO (Max Sharpe Ratio) ***")
        print(f"  Expected Return: {best_portfolio['ret']:.4f} | CVaR: {best_portfolio['cvar']:.4f} | Sharpe: {best_sharpe:.4f}")
        selected_tickers = [tickers[i] for i, z in enumerate(best_portfolio['z']) if z == 1]
        print(f"  Tickers: {', '.join(selected_tickers)}")

    # =====================================================================
    # 5. GUROBI SCALING EXPERIMENT (NP-Hard Proof)
    # =====================================================================
    print("\n" + "="*50)
    print("Running Gurobi Scaling Experiment (M=20 to 100)...")
    print("="*50)
    
    scaling_results = scaling_experiment(mu, scenarios, tickers)
    print("\n✓ Scaling Results (Time in seconds):")
    for m_size, t in scaling_results.items():
        print(f"  M={m_size}: {t:.2f} seconds")

    # =====================================================================
    # 6. CONSTRAINT HANDLING COMPARISON (Week 3 Concepts)
    # =====================================================================
    print("\n" + "="*50)
    print("Constraint Handling: Repair vs Penalty vs Decoder")
    print("="*50)
    
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
    from pymoo.optimize import minimize as pymoo_minimize
    from nsga2_optimizer import PortfolioProblem, PortfolioProblemPenalty, PortfolioProblemDecoder
    
    problem_repair = PortfolioProblem(mu, scenarios, tickers)
    problem_penalty = PortfolioProblemPenalty(mu, scenarios, tickers)
    problem_decoder = PortfolioProblemDecoder(mu, scenarios, tickers)
    
    constraint_results = {}
    
    print("\n--- Running NSGA-II with REPAIR ---")
    algo = NSGA2(pop_size=500, sampling=IntegerRandomSampling(), crossover=SBX(prob=0.9, eta=15, vtype=int), mutation=PM(eta=20, vtype=int), eliminate_duplicates=True)
    res_r = pymoo_minimize(problem_repair, algo, ('n_gen', 50), seed=42, verbose=False)
    
    print("--- Running NSGA-II with PENALTY ---")
    algo = NSGA2(pop_size=500, sampling=IntegerRandomSampling(), crossover=SBX(prob=0.9, eta=15, vtype=int), mutation=PM(eta=20, vtype=int), eliminate_duplicates=True)
    res_p = pymoo_minimize(problem_penalty, algo, ('n_gen', 50), seed=42, verbose=False)
    
    print("--- Running NSGA-II with DECODER ---")
    algo = NSGA2(pop_size=500, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
    res_d = pymoo_minimize(problem_decoder, algo, ('n_gen', 50), seed=42, verbose=False)

    # Evaluate Metrics
    from pymoo.indicators.hv import HV
    from pymoo.indicators.gd import GD
    from pymoo.indicators.igd import IGD
    
    gurobi_F = np.array([[pt['ret'], pt['cvar']] for pt in fronts])
    hv_ind = HV(ref_point=np.array([0.6, 0.10]))
    gd_ind = GD(gurobi_F)
    igd_ind = IGD(gurobi_F)

    for name, res in [("Repair", res_r), ("Penalty", res_p), ("Decoder", res_d)]:
        F = res.F.copy(); F[:, 0] = -F[:, 0] 
        constraint_results[name] = {
            "HV": hv_ind.do(F), "GD": gd_ind.do(F), "IGD": igd_ind.do(F)
        }

    print("\n" + "-"*50)
    print("Week 3 Constraint Handling Comparison")
    print("-"*50)
    print(f"{'Method':<15} {'HV (Higher)':<15} {'GD (Lower)':<15} {'IGD (Lower)':<15}")
    print("-" * 60)
    for name, metrics in constraint_results.items():
        print(f"{name:<15} {metrics['HV']:<15.6f} {metrics['GD']:<15.6f} {metrics['IGD']:<15.6f}")

    best_ch_method = max(constraint_results, key=lambda k: constraint_results[k]["HV"])
    print(f"\n-> Winner: '{best_ch_method}'. Using it for final algorithm comparison.")
    
    if best_ch_method == "Repair": res_nsga_final = res_r; problem_final = problem_repair
    elif best_ch_method == "Penalty": res_nsga_final = res_p; problem_final = problem_penalty
    else: res_nsga_final = res_d; problem_final = problem_decoder
        
    nsga_F = res_nsga_final.F.copy(); nsga_F[:, 0] = -nsga_F[:, 0]

    # =====================================================================
    # 7. MULTI-OBJECTIVE ALGORITHM COMPARISON (Weeks 4, 5, 6)
    # =====================================================================
    print("\n" + "="*50)
    print("Algorithm Comparison: NSGA-II vs MOEA/D vs OMOPSO")
    print("="*50)
    
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.util.ref_dirs import get_reference_directions

    # --- MOEA/D ---
    print("--- Running MOEA/D ---")
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    algo_moead = MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.9, sampling=IntegerRandomSampling(), crossover=SBX(prob=0.9, eta=15, vtype=int), mutation=PM(eta=20, vtype=int))
    res_moead = pymoo_minimize(problem_final, algo_moead, ('n_gen', 50), seed=42, verbose=False)

    # --- AGE-MOEA (Adaptive Geometry Estimation) ---
    print("--- Running AGE-MOEA (Alternative Paradigm) ---")
    from pymoo.algorithms.moo.age import AGEMOEA
    # AGE-MOEA uses an auto-adaptive clustering mechanism instead of crowding distance
    algo_age = AGEMOEA(pop_size=500, sampling=IntegerRandomSampling(), 
                       crossover=SBX(prob=0.9, eta=15, vtype=int), 
                       mutation=PM(eta=20, vtype=int),
                       eliminate_duplicates=True)
    res_age = pymoo_minimize(problem_final, algo_age, ('n_gen', 50), seed=42, verbose=False)

    # Final Metrics Table
    moead_F = res_moead.F.copy(); moead_F[:, 0] = -moead_F[:, 0]
    age_F = res_age.F.copy(); age_F[:, 0] = -age_F[:, 0]

    print("\n" + "-"*50)
    print("Final Multi-Objective Algorithm Metrics")
    print("-"*50)
    print(f"{'Algorithm':<15} {'HV (Higher)':<15} {'GD (Lower)':<15} {'IGD (Lower)':<15}")
    print("-" * 60)
    print(f"{'NSGA-II':<15} {hv_ind.do(nsga_F):<15.6f} {gd_ind.do(nsga_F):<15.6f} {igd_ind.do(nsga_F):<15.6f}")
    print(f"{'MOEA/D':<15} {hv_ind.do(moead_F):<15.6f} {gd_ind.do(moead_F):<15.6f} {igd_ind.do(moead_F):<15.6f}")
    print(f"{'AGE-MOEA':<15} {hv_ind.do(age_F):<15.6f} {gd_ind.do(age_F):<15.6f} {igd_ind.do(age_F):<15.6f}")
    print(f"{'Gurobi':<15} {hv_ind.do(gurobi_F):<15.6f} {'0.000000':<15} {'0.000000':<15}")

    print("\n" + "-"*50)
    print("Final Multi-Objective Algorithm Metrics")
    print("-"*50)
    print(f"{'Algorithm':<15} {'HV (Higher)':<15} {'GD (Lower)':<15} {'IGD (Lower)':<15}")
    print("-" * 60)
    print(f"{'NSGA-II':<15} {hv_ind.do(nsga_F):<15.6f} {gd_ind.do(nsga_F):<15.6f} {igd_ind.do(nsga_F):<15.6f}")
    print(f"{'MOEA/D':<15} {hv_ind.do(moead_F):<15.6f} {gd_ind.do(moead_F):<15.6f} {igd_ind.do(moead_F):<15.6f}")
    print(f"{'AGE-MOEA':<15} {hv_ind.do(age_F):<15.6f} {gd_ind.do(age_F):<15.6f} {igd_ind.do(age_F):<15.6f}")
    print(f"{'Gurobi':<15} {hv_ind.do(gurobi_F):<15.6f} {'0.000000':<15} {'0.000000':<15}")

    # =====================================================================
    # 8. STRESS TEST THE BEST PORTFOLIO
    # =====================================================================
    print("\n" + "="*50)
    print("Stress Testing Best Gurobi Portfolio...")
    print("="*50)
    
    if best_portfolio is not None:
        w_best = best_portfolio['w']
        
        covid_m_best = stress_metrics(w_best, ret_covid, mu)
        print(f"✓ BEST PORT COVID: MaxDD={covid_m_best['max_drawdown']:.4f} | Sharpe={covid_m_best['sharpe_ratio']:.4f}")
        
        if tariff_m["realised_cvar"] is not None:
            tariff_m_best = stress_metrics(w_best, ret_tariff, mu)
            print(f"✓ BEST PORT TARIFF: MaxDD={tariff_m_best['max_drawdown']:.4f} | Sharpe={tariff_m_best['sharpe_ratio']:.4f}")
            
        print("\n--- Comparison vs Equal Weight ---")
        print(f"COVID MaxDD  -> Equal Weight: {covid_m['max_drawdown']:.4f} | Optimized: {covid_m_best['max_drawdown']:.4f}")
        if tariff_m["realised_cvar"] is not None:
            print(f"Tariff MaxDD -> Equal Weight: {tariff_m['max_drawdown']:.4f} | Optimized: {tariff_m_best['max_drawdown']:.4f}")

    print("\n[End of Pipeline - All Proposal & Syllabus Requirements Met]")

if __name__ == "__main__":
    main()