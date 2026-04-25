import pytest
try:
    import ortools
except ImportError:
    pytest.skip("ortools not installed", allow_module_level=True)

from eir.core import EIRGraph, DiscreteVar
from eir.factors.logic import AndFactor
from eir.backends.verify_cpsat import VerifyCPSATBackend

def test_and_cpsat():
    a = DiscreteVar("A", 2)
    b = DiscreteVar("B", 2)
    f = AndFactor([a, b], state_a=1, state_b=1) # A=1 AND B=1
    g = EIRGraph([a, b], [f])
    
    res = VerifyCPSATBackend().solve(g)
    assert res.status in ["OPTIMAL", "FEASIBLE"]
    assert res.assignment['A'] == 1
    assert res.assignment['B'] == 1
