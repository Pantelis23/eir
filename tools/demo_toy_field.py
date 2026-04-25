import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eir.core import EIRGraph, DiscreteVar
from eir.factors.logic import XorFactor, AndFactor
from eir.backends.field import FieldBackend

def main():
    print("--- Demo: Field Backend (Toy Logic) ---")
    
    # Problem: A XOR B = 1, A AND C = 1
    # Solution: A=1, B=0, C=1
    
    a = DiscreteVar("A", 2)
    b = DiscreteVar("B", 2)
    c = DiscreteVar("C", 2)
    
    f1 = XorFactor([a, b], target=1, weight=5.0)
    f2 = AndFactor([a, c], state_a=1, state_b=1, weight=5.0)
    
    graph = EIRGraph([a, b, c], [f1, f2])
    
    backend = FieldBackend(steps=2000, lr=0.1, noise_start=0.5, noise_end=0.01)
    result = backend.solve(graph)
    
    print(f"Energy: {result.energy}")
    print(f"Assignment: {result.assignment}")
    print(f"Converged: {result.converged}")
    
    # Check correctness
    val_a = result.assignment['A']
    val_b = result.assignment['B']
    val_c = result.assignment['C']
    
    if val_a == 1 and val_b == 0 and val_c == 1:
        print("SUCCESS: Found correct solution.")
    else:
        print("FAILURE: Did not find optimal solution.")

if __name__ == "__main__":
    main()
