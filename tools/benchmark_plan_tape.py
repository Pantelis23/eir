import numpy as np
import time
import sys
import itertools
import os
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import planners.eir_plan_tape as P
from planners.eir_plan_tape import EIRPlanTapePlanner, EIR_AVAILABLE

def benchmark():
    src = inspect.getsource(P)
    if "itertools" in src or "product(" in src or "repeat=" in src:
        print("CHEAT: planner enumerates sequences via itertools/product")
        sys.exit(1)
        
    if not EIR_AVAILABLE:
        print("SKIP: EIR not available")
        sys.exit(0)
        
    print("Starting PLAN TAPE CHANCE-CVAR Benchmark...")
    np.random.seed(0)
    
    H = 7
    K = 6
    z_dim = 2
    
    # Dynamics
    f_true_table = {}
    for k in range(K):
        theta = (k / K) * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [s, c]]) * 0.9 
        b = np.random.randn(z_dim) * 0.1
        f_true_table[k] = (A, b)
        
    f_model_table = {}
    for k in range(K):
        A_true, b_true = f_true_table[k]
        A_model = A_true + np.random.randn(z_dim, z_dim) * 0.03
        b_model = b_true + np.random.randn(z_dim) * 0.02
        f_model_table[k] = (A_model, b_model)
        
    r_table = {}
    for k in range(K):
        r_table[k] = (np.random.randn(z_dim), np.random.randn() * 0.1)
        
    z0_obs = np.array([1.0, 0.0])
    sigma = 0.15
    num_particles = 32
    particles_eps = np.random.randn(num_particles, z_dim)
    
    # Constraints
    constraints = {2: [3], 6: [5]} 
    
    beta = 10.0
    lambda_dyn = 2.0
    lambda_risk = 5.0
    y_max = 0.8
    eps_forbid = 0.05
    cvar_q = 0.10
    lambda_cvar = 0.5
    
    print(f"Brute forcing (DFS) {K}^{H} sequences with particles...")
    best_seq = None
    best_J = float('inf')
    
    # Precompute
    F_A = np.zeros((K, z_dim, z_dim))
    F_b = np.zeros((K, z_dim))
    M_A = np.zeros((K, z_dim, z_dim))
    M_b = np.zeros((K, z_dim))
    R_w = np.zeros((K, z_dim))
    R_c = np.zeros(K)
    
    for k in range(K):
        F_A[k], F_b[k] = f_true_table[k]
        M_A[k], M_b[k] = f_model_table[k]
        R_w[k], R_c[k] = r_table[k]

    def get_forbidden_fraction(t, z_batch):
        # z_batch: (P, z_dim)
        fixed_forbidden = set(constraints.get(t, []))
        x = z_batch[:, 0]
        y = z_batch[:, 1]
        
        fractions = np.zeros(K)
        mask_0 = (y > 0.5)
        mask_1 = (y < -0.5)
        mask_2 = (np.abs(x) > 1.2)
        
        P = len(z_batch)
        fractions[0] = np.sum(mask_0) / P
        fractions[1] = np.sum(mask_1) / P
        fractions[2] = np.sum(mask_2) / P
        
        for f in fixed_forbidden:
            fractions[f] = 1.0
        return fractions

    def compute_cvar(values, q):
        if q <= 0: return 0.0
        if q >= 1.0: return np.mean(values)
        k = int(np.ceil(q * len(values)))
        if k == 0: k = 1
        top_vals = np.sort(values)[-k:]
        return np.mean(top_vals)

    # Particles init
    z0_particles = z0_obs + sigma * particles_eps

    def dfs(t, z_curr, Ep_acc, seq_acc):
        nonlocal best_J, best_seq
        
        if t == H:
            mean_E = np.mean(Ep_acc)
            cvar_E = compute_cvar(Ep_acc, cvar_q)
            J = mean_E + lambda_cvar * cvar_E
            if J < best_J:
                best_J = J
                best_seq = list(seq_acc)
            return

        fracs = get_forbidden_fraction(t, z_curr)
        
        for k in range(K):
            if fracs[k] > eps_forbid: continue
            
            # Vectorized step
            r_p = np.sum(R_w[k] * z_curr, axis=1) + R_c[k]
            
            z_pred = (M_A[k] @ z_curr[:, :, None]).squeeze(2) + M_b[k]
            z_next = (F_A[k] @ z_curr[:, :, None]).squeeze(2) + F_b[k]
            
            diff = z_next - z_pred
            c_cost_p = np.sum(diff**2, axis=1)
            
            y_next = z_next[:, 1]
            risk_p = np.maximum(0.0, np.abs(y_next) - y_max)**2
            
            step_E_p = lambda_dyn * c_cost_p + lambda_risk * risk_p - beta * r_p
            
            dfs(t+1, z_next, Ep_acc + step_E_p, seq_acc + [k])

    t_start = time.perf_counter()
    dfs(0, z0_particles, np.zeros(num_particles), [])
    t_end = time.perf_counter()
    print(f"Brute Force done in {t_end - t_start:.4f}s. Best J: {best_J:.4f}")
    
    # Planner
    planner = EIRPlanTapePlanner(H=H, z_dim=z_dim, K=K, beta=beta, lambda_dyn=lambda_dyn, 
                                 lambda_risk=lambda_risk, y_max=y_max, beam_width=32,
                                 eps_forbid=eps_forbid, cvar_q=cvar_q, lambda_cvar=lambda_cvar)
    
    start_t = time.perf_counter()
    res = planner.plan(z0_obs, f_true_table, f_model_table, r_table, constraints, 
                       sigma=sigma, particles_eps=particles_eps, num_particles=num_particles)
    end_t = time.perf_counter()
    
    duration = end_t - start_t
    print(f"Planner Time: {duration:.4f}s")
    print(f"Planner Seq: {res['actions']}, J: {res['energy']:.4f}")
    print(f"Eval Count: {res.get('eval_count', 'N/A')}")
    
    if res.get('eval_count', 999999) > 60000:
        print(f"FAIL: Eval count too high: {res['eval_count']} > 60000")
        sys.exit(1)
        
    p_seq = res['actions']
    
    # Validate result J
    z_curr = z0_particles.copy()
    Ep_acc = np.zeros(num_particles)
    
    for t, a in enumerate(p_seq):
        fracs = get_forbidden_fraction(t, z_curr)
        if fracs[a] > eps_forbid:
            print(f"FAIL: Planner violated constraint at t={t}, action={a}, frac={fracs[a]}")
            sys.exit(1)
            
        r_p = np.sum(R_w[a] * z_curr, axis=1) + R_c[a]
        z_pred = (M_A[a] @ z_curr[:, :, None]).squeeze(2) + M_b[a]
        z_next = (F_A[a] @ z_curr[:, :, None]).squeeze(2) + F_b[a]
        diff = z_next - z_pred
        c_cost_p = np.sum(diff**2, axis=1)
        y_next = z_next[:, 1]
        risk_p = np.maximum(0.0, np.abs(y_next) - y_max)**2
        
        Ep_acc += lambda_dyn * c_cost_p + lambda_risk * risk_p - beta * r_p
        z_curr = z_next
        
    mean_E = np.mean(Ep_acc)
    cvar_E = compute_cvar(Ep_acc, cvar_q)
    pJ = mean_E + lambda_cvar * cvar_E
    
    print(f"Planner True J: {pJ:.4f}")
    
    diff = abs(pJ - best_J)
    tol = max(0.05, 0.01 * abs(best_J))
    
    if diff > tol:
        print(f"FAIL: J difference too high. {diff:.4f} > {tol:.4f}")
        sys.exit(1)
        
    if duration > 2.5:
        print(f"FAIL: Too slow. {duration:.4f}s > 2.5s")
        sys.exit(1)
        
    print("PLAN TAPE CHANCE-CVAR: PASS")

if __name__ == "__main__":
    benchmark()
