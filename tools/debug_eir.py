import numpy as np
import logging
from eir.core import EIRGraph, DiscreteVar
from eir.factors.bias import ActionBiasFactor
from eir.factors.onehot import OneHotFactor
from eir.backends.field import FieldBackend

def debug_case1():
    # Replicate Case 1 logic
    logits = np.array([0.0, 10.0, 0.0, 5.0, 0.0])
    action_mask = np.array([1, 0, 1, 1, 1], dtype=bool)
    
    # Logic from EIRActionSelector
    masked_logits = logits.copy()
    masked_logits[~action_mask] = -1e9
    
    valid = masked_logits > -1e8
    
    # Cost calc
    l = logits / 1.0
    valid_l = l[valid]
    max_val = np.max(valid_l)
    l_shift = l - max_val
    
    invalid_penalty = 1e6
    costs = (-l_shift) + (invalid_penalty * (~valid).astype(float))
    
    print("Logits:", logits)
    print("Valid:", valid)
    print("Max val (valid):", max_val)
    print("L shift:", l_shift)
    print("Costs:", costs)
    
    # Graph
    action_var = DiscreteVar("Action", 5)
    onehot = OneHotFactor([action_var], weight=0.01)
    bias = ActionBiasFactor([action_var], costs=costs.tolist(), weight=1.0)
    
    graph = EIRGraph([action_var], [onehot, bias])
    
    # Backend
    backend = FieldBackend(steps=1000, num_starts=5, lr=0.1, noise_start=0.5)
    res = backend.solve(graph)
    
    print("Result Assignment:", res.assignment)
    print("Result Energy:", res.energy)
    
    choice = int(res.assignment['Action'])
    print(f"Choice: {choice}")
    
    if choice == 3:
        print("DEBUG SUCCESS")
    else:
        print("DEBUG FAIL")

if __name__ == "__main__":
    debug_case1()
