import numpy as np
import pytest
from eir.core import EIRGraph, DiscreteVar
from eir.factors.bias import ActionBiasFactor
from eir.backends.field import FieldBackend
from eir.serialize import serialize_graph, deserialize_graph

def test_bias_factor_roundtrip():
    """Test serialization of ActionBiasFactor."""
    a = DiscreteVar("A", 3)
    f = ActionBiasFactor([a], costs=[0.1, 0.2, 0.3])
    g = EIRGraph([a], [f])
    
    s = serialize_graph(g)
    g2 = deserialize_graph(s)
    
    f2 = g2.factors[0]
    assert isinstance(f2, ActionBiasFactor)
    assert np.allclose(f2.costs, [0.1, 0.2, 0.3])

def test_bias_selection():
    """Test that ActionBiasFactor drives selection."""
    a = DiscreteVar("A", 3)
    # Action 1 is cheapest
    f = ActionBiasFactor([a], costs=[10.0, 0.0, 5.0])
    g = EIRGraph([a], [f])
    
    res = FieldBackend(steps=500, num_starts=3).solve(g)
    assert res.assignment['A'] == 1
    assert res.energy < 0.1
