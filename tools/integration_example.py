import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from planners.eir_planner import EIRActionSelector, EIR_AVAILABLE

def main():
    print("Starting Integration Example...")
    
    # 1. Integration Logic
    B, A = 5, 10
    logits = np.random.randn(B, A)
    # Simple masking
    am = np.ones((B, A))
    am[:, 0] = 0 # Mask action 0
    
    # Initialize Selector
    # Reduce steps for speed in example
    selector = EIRActionSelector(steps=200, num_starts=2)
    
    # Call select
    try:
        actions = selector.select(
            logits, 
            action_dim=A, 
            inventory_dim=0, 
            action_mask=am
        )
        print(f"Selected actions shape: {actions.shape}")
        print(f"Sample actions: {actions}")
        if isinstance(actions, np.ndarray) and actions.shape == (B,):
            print("INTEGRATION EXAMPLE: PASS")
        else:
            print("INTEGRATION EXAMPLE: FAIL (Bad shape/type)")
            sys.exit(1)
            
    except Exception as e:
        print(f"INTEGRATION EXAMPLE: FAIL (Exception: {e})")
        sys.exit(1)

    # 2. Performance Sanity Check
    print("\nStarting Performance Check...")
    B_perf = 32
    A_perf = 64
    N_runs = 5
    
    logits_p = np.random.randn(B_perf, A_perf)
    am_p = np.ones((B_perf, A_perf))
    
    # Use lightweight config for perf test to prove architecture throughput
    # (Python loop overhead in FieldBackend is high)
    perf_selector = EIRActionSelector(steps=50, num_starts=1)
    
    start_t = time.time()
    for _ in range(N_runs):
        _ = perf_selector.select(logits_p, action_dim=A_perf, inventory_dim=0, action_mask=am_p)
    end_t = time.time()
    
    duration = end_t - start_t
    print(f"Duration for {N_runs} runs: {duration:.4f}s")
    
    limit = 10.0 if EIR_AVAILABLE else 0.5
    mode = "EIR" if EIR_AVAILABLE else "ARGMAX"
    
    if duration < limit:
        print(f"PERF: PASS (mode={mode}, seconds={duration:.4f})")
    else:
        print(f"PERF: FAIL (mode={mode}, seconds={duration:.4f} > {limit})")
        # Don't fail the build for perf yet, just warn, unless strictly required. 
        # The prompt says "If perf fails, reduce ...". 
        # I will exit 1 if it fails to force tuning.
        sys.exit(1)

if __name__ == "__main__":
    main()
