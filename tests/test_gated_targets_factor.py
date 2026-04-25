import numpy as np
import pytest
from eir.core import EIRGraph, DiscreteVar, ContinuousVar
from eir.factors.gated import GatedTargetsFactor
from eir.backends.field import FieldBackend
from eir.serialize import serialize_graph, deserialize_graph

def test_gated_targets_selection():
    """
    Test that GatedTargetsFactor correctly selects the action with the lowest
    total cost (regularization + distance).
    """
    # Action 0: Target 10.0 (High Reg Cost)
    # Action 1: Target 0.5 (Low Reg Cost)
    targets = [np.array([10.0]), np.array([0.5])]
    
    action = DiscreteVar("A", 2)
    p0 = ContinuousVar("P0", 1)
    p1 = ContinuousVar("P1", 1)
    
    # lam=0.1 -> Cost0 approx 0.1 * 100 = 10. Cost1 approx 0.1 * 0.25 = 0.025
    factor = GatedTargetsFactor([action, p0, p1], targets, lam=0.1)
    graph = EIRGraph([action, p0, p1], [factor])
    
    backend = FieldBackend(steps=1000, num_starts=5, noise_start=0.5)
    res = backend.solve(graph)
    
    assert res.assignment['A'] == 1
    assert np.allclose(res.assignment['P1'], 0.5, atol=0.2)
    # P0 should be near 0 because it's inactive and regularized
    assert np.allclose(res.assignment['P0'], 0.0, atol=0.2) 

def test_gated_factor_serialization():
    """Test round-trip serialization of GatedTargetsFactor."""
    targets = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    action = DiscreteVar("A", 2)
    p0 = ContinuousVar("P0", 2)
    p1 = ContinuousVar("P1", 2)
    
    factor = GatedTargetsFactor([action, p0, p1], targets, lam=0.05)
    graph = EIRGraph([action, p0, p1], [factor])
    
    json_str = serialize_graph(graph)
    g2 = deserialize_graph(json_str)
    
    assert len(g2.factors) == 1
    f2 = g2.factors[0]
    assert isinstance(f2, GatedTargetsFactor)
    assert f2.lam == 0.05
    assert len(f2.targets) == 2
    assert np.allclose(f2.targets[0], targets[0])
    assert np.allclose(f2.targets[1], targets[1])
