import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eir.core import EIRGraph, DiscreteVar, ContinuousVar
from eir.factors.gated import GatedTargetsFactor
from eir.factors.bias import ActionBiasFactor
from eir.backends.field import FieldBackend

def main():
    print("--- Demo: Mixed Discrete-Continuous Planning ---")
    np.random.seed(42) # Ensure reproducibility
    
    # 1. Action: Choose 0, 1, 2
    action = DiscreteVar("Action", 3)
    
    # 2. Parameters for each action
    p0 = ContinuousVar("P0", 1)
    p1 = ContinuousVar("P1", 1)
    p2 = ContinuousVar("P2", 1)
    
    # 3. Factor connecting them
    # Targets: P0->0.5, P1->0.9, P2->-0.1
    # We use realistic targets now (no hack).
    targets = [np.array([0.5]), np.array([0.9]), np.array([-0.1])]
    
    # Gated Factor: handles parameter targets + regularization
    # Lower lam because we don't rely on it for action selection anymore.
    gated_factor = GatedTargetsFactor(
        [action, p0, p1, p2], 
        targets, 
        lam=0.01 
    )
    
    # Bias Factor: Explicitly prefer Action 1
    # Cost0: 0.1, Cost1: 0.0, Cost2: 0.5
    bias_factor = ActionBiasFactor(
        [action],
        costs=[0.1, 0.0, 0.5]
    )
    
    graph = EIRGraph([action, p0, p1, p2], [gated_factor, bias_factor])
    
    # 4. Solve
    backend = FieldBackend(steps=2000, lr=0.05, noise_start=0.5, noise_end=0.01, num_starts=5)
    result = backend.solve(graph)
    
    print(f"Energy: {result.energy}")
    print(f"Assignment: {result.assignment}")
    
    a_val = result.assignment['Action']
    p1_val = result.assignment['P1']
    
    # Optimal: Action=1 
    if a_val == 1 and np.abs(p1_val - 0.9) < 0.1:
        print("SUCCESS: Selected Action 1 and optimized P1.")
    else:
        print("FAILURE: Suboptimal result.")
        
    # Check bounded regression
    p0_val = result.assignment['P0']
    p2_val = result.assignment['P2']
    
    # They should be bounded near 0
    if np.abs(p0_val) < 1.0 and np.abs(p2_val) < 1.0:
        print("SUCCESS: Unused parameters P0/P2 are bounded.")
    else:
        print(f"WARNING: Unused parameters value: P0={p0_val}, P2={p2_val}")

if __name__ == "__main__":
    main()