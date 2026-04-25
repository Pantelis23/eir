import numpy as np
import sys
import os
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import planners.eir_plan_tape as P
import planners.eir_belief_mpc as B_MPC
from planners.eir_belief_mpc import EIRBeliefMPC

def benchmark():
    # Non-cheat check
    for mod in [P, B_MPC]:
        src = inspect.getsource(mod)
        if "itertools" in src or "product(" in src or "repeat=" in src:
            print(f"CHEAT: {mod.__name__} enumerates sequences")
            sys.exit(1)
            
    print("Starting BELIEF MPC Benchmark...")
    
    # Setup
    seed = 0
    rng = np.random.default_rng(seed)
    
    K = 6
    z_dim = 2
    
    # Dynamics
    f_true_table = {}
    for k in range(K):
        theta = (k / K) * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [s, c]]) * 0.9 
        b = rng.normal(size=z_dim) * 0.1
        f_true_table[k] = (A, b)
        
    f_model_table = {}
    for k in range(K):
        A_true, b_true = f_true_table[k]
        A_model = A_true + rng.normal(size=(z_dim, z_dim)) * 0.03
        b_model = b_true + rng.normal(size=z_dim) * 0.02
        f_model_table[k] = (A_model, b_model)
        
    r_table = {}
    for k in range(K):
        r_table[k] = (rng.normal(size=z_dim), rng.normal() * 0.1)
        
    constraints_fixed = {2: [3], 6: [5]}
    
    # Params
    params = {
        'K': K, 'z_dim': z_dim,
        'beta': 10.0, 'lambda_dyn': 2.0, 'lambda_risk': 5.0, 'y_max': 0.8,
        'eps_forbid': 0.05, 'cvar_q': 0.10, 'lambda_cvar': 0.5,
        'beam_width': 32, 'local_search_candidates': 10, 'local_search_passes': 2
    }
    
    z0_true = np.array([1.0, 0.0])
    obs_sigma = 0.20
    z0_obs = z0_true + obs_sigma * rng.normal(size=2)
    
    # Run Belief MPC
    mpc = EIRBeliefMPC(planner_params=params)
    res_belief = mpc.run_episode(
        z0_true, z0_obs, f_true_table, f_model_table, r_table, constraints_fixed,
        T=10, H=5, P=64, obs_sigma=obs_sigma, proc_sigma=0.03, seed=0
    )
    
    # Run Baseline (Static Belief - disable filter)
    res_baseline = mpc.run_episode(
        z0_true, z0_obs, f_true_table, f_model_table, r_table, constraints_fixed,
        T=10, H=5, P=64, obs_sigma=obs_sigma, proc_sigma=0.03, seed=0,
        use_filter=False
    )
    
    # Print results
    print(f"Belief MPC Total Cost: {res_belief['total_cost']:.4f}")
    print(f"Fixed Violations: {res_belief['violations_fixed']}")
    print(f"State Violations: {res_belief['violations_state']}")
    
    print(f"Baseline MPC Total Cost: {res_baseline['total_cost']:.4f}")
    print(f"Fixed Violations: {res_baseline['violations_fixed']}")
    print(f"State Violations: {res_baseline['violations_state']}")
    
    # Assertions
    # 1. No fixed violations
    if res_belief['violations_fixed'] > 0 or res_baseline['violations_fixed'] > 0:
        print("FAIL: Fixed constraints violated")
        sys.exit(1)
        
    # 2. State violations <= 1
    if res_belief['violations_state'] > 1:
        print(f"FAIL: Belief MPC state violations {res_belief['violations_state']} > 1")
        sys.exit(1)
        
    if res_baseline['violations_state'] > 1:
        print(f"FAIL: Baseline MPC state violations {res_baseline['violations_state']} > 1")
        sys.exit(1)
        
    # 3. Belief improvement >= 2.0
    improvement = res_baseline['total_cost'] - res_belief['total_cost']
    print(f"Improvement: {improvement:.4f}")
    
    if improvement < 2.0:
        print(f"FAIL: Improvement {improvement:.4f} < 2.0")
        sys.exit(1)
        
    print("BELIEF MPC: PASS")

if __name__ == "__main__":
    benchmark()