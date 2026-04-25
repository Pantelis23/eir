from eir.core import EIRGraph, DiscreteVar
from eir.factors.logic import XorFactor
from eir.backends.field import FieldBackend

def test_xor_field():
    a = DiscreteVar("A", 2)
    b = DiscreteVar("B", 2)
    f = XorFactor([a, b], target=1) # A != B
    g = EIRGraph([a, b], [f])
    
    # Solve multiple times to ensure stability
    for _ in range(3):
        res = FieldBackend(steps=500).solve(g)
        assert res.assignment['A'] != res.assignment['B']
