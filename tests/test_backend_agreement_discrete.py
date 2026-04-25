import pytest
try:
    import ortools
except ImportError:
    pytest.skip("ortools not installed", allow_module_level=True)

from eir.core import EIRGraph, DiscreteVar
from eir.factors.logic import XorFactor
from eir.backends.field import FieldBackend
from eir.backends.verify_cpsat import VerifyCPSATBackend

def test_agreement():
    # Problem: A XOR B.
    # Both should find a valid solution (0,1) or (1,0).
    a = DiscreteVar("A", 2)
    b = DiscreteVar("B", 2)
    f = XorFactor([a, b], target=1)
    g = EIRGraph([a, b], [f])
    
    res_field = FieldBackend(steps=1000).solve(g)
    res_verify = VerifyCPSATBackend().solve(g)
    
    # Check validity
    field_ok = (res_field.assignment['A'] != res_field.assignment['B'])
    verify_ok = (res_verify.assignment['A'] != res_verify.assignment['B'])
    
    assert field_ok
    assert verify_ok
