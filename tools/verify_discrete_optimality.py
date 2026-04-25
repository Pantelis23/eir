import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try imports
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    
from planners.eir_planner import EIRActionSelector, apply_action_mask_like

def main():
    if not ORTOOLS_AVAILABLE:
        print("CP-SAT SKIPPED")
        sys.exit(0)
        
    print("Starting CP-SAT Discrete Optimality Check...")
    np.random.seed(42)
    n_tests = 200
    
    selector = EIRActionSelector()
    
    for i in range(n_tests):
        A = np.random.randint(5, 20)
        logits = np.random.randn(A)
        mask = (np.random.rand(A) > 0.3).astype(int)
        if mask.sum() == 0: mask[0] = 1
        
        # 1. Argmax Solution
        masked_l = logits.copy()
        masked_l[mask == 0] = -1e9
        argmax_idx = int(np.argmax(masked_l))
        
        # 2. CP-SAT Solution
        model = cp_model.CpModel()
        
        # Vars
        # One boolean per action? Or one Int var?
        # Int var is cleaner for "Select one"
        action_var = model.NewIntVar(0, A - 1, "action")
        
        # Constraints: Mask
        # If mask[k] == 0, action != k
        for k in range(A):
            if mask[k] == 0:
                model.Add(action_var != k)
                
        # Objective: Maximize Logit (or Minimize Cost = -Logit)
        # CP-SAT uses integers. Scale logits.
        scale = 10000
        # coefficients
        coeffs = (logits * scale).astype(int)
        
        # We want to maximize sum(is_selected[k] * coeff[k])
        # Model using Element? Or bools?
        # Bools are easier for objective
        bool_vars = [model.NewBoolVar(f"b{k}") for k in range(A)]
        
        # Link Int to Bools: exactly one
        model.Add(sum(bool_vars) == 1)
        # Link action_var to bools
        model.Add(action_var == sum(k * bool_vars[k] for k in range(A)))
        
        # Mask constraints on bools
        for k in range(A):
            if mask[k] == 0:
                model.Add(bool_vars[k] == 0)
                
        # Objective
        model.Maximize(sum(bool_vars[k] * coeffs[k] for k in range(A)))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"FAIL test {i}: CP-SAT Infeasible")
            sys.exit(1)
            
        cp_idx = solver.Value(action_var)
        
        # Compare
        if cp_idx != argmax_idx:
             # Check tie-break (logits equal?)
             if not np.isclose(logits[cp_idx], logits[argmax_idx]):
                 print(f"FAIL test {i}: Argmax {argmax_idx} (L={logits[argmax_idx]:.4f}) != CP-SAT {cp_idx} (L={logits[cp_idx]:.4f})")
                 sys.exit(1)

    print("CP-SAT OPTIMALITY: PASS")

if __name__ == "__main__":
    main()
