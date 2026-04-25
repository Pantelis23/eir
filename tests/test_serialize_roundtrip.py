import json
from eir.core import EIRGraph, DiscreteVar, ContinuousVar
from eir.factors.quadratic import QuadraticFactor
from eir.serialize import serialize_graph, deserialize_graph

def test_roundtrip():
    v1 = DiscreteVar("D1", 5)
    v2 = ContinuousVar("C1", 3)
    f1 = QuadraticFactor([v2], target=[1.0, 2.0, 3.0])
    
    g = EIRGraph([v1, v2], [f1])
    
    s = serialize_graph(g)
    g2 = deserialize_graph(s)
    
    assert len(g2.vars) == 2
    assert g2.get_var("D1").n == 5
    assert len(g2.factors) == 1
    assert isinstance(g2.factors[0], QuadraticFactor)
    assert g2.factors[0].target[1] == 2.0
