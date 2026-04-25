import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eir.core import EIRGraph, DiscreteVar, Factor
from eir.factors.logic import XorFactor, ImplicationFactor
from eir.backends.verify_cpsat import VerifyCPSATBackend

def main():
    print("--- Demo: Verify CP-SAT Backend ---")
    
    try:
        import ortools
    except ImportError:
        print("OR-Tools not installed. Skipping.")
        return

    # Problem: A -> B, A=1. Therefore B must be 1.
    a = DiscreteVar("A", 2)
    b = DiscreteVar("B", 2)
    
    class ForceOne(Factor):
        def eval(self, assign):
            return 1.0 - assign[self.scope[0].name] if isinstance(assign[self.scope[0].name], float) else (0.0 if assign[self.scope[0].name] == 1 else 1.0)
        def compile_cpsat(self, model, varmap):
            model.Add(varmap[self.scope[0].name] == 1)
            
    f1 = ImplicationFactor([a, b], state_a=1, state_b=1)
    f2 = ForceOne([a])
    
    graph = EIRGraph([a, b], [f1, f2])
    
    backend = VerifyCPSATBackend(strict=True)
    result = backend.solve(graph)
    
    print(f"Status: {result.status}")
    print(f"Assignment: {result.assignment}")
    
    if result.assignment and result.assignment['A'] == 1 and result.assignment['B'] == 1:
        print("SUCCESS")
    else:
        print("FAILURE")

if __name__ == "__main__":
    main()