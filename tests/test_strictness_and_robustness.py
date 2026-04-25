import pytest
import json
import numpy as np
from eir.core import EIRGraph, DiscreteVar, Factor
from eir.serialize import serialize_graph, deserialize_graph
from eir.backends.verify_cpsat import VerifyCPSATBackend
from eir.factors.logic import ImplicationFactor

def test_strict_deserialize_unknown_factor():
    """Test that deserialize_graph raises ValueError on unknown factor when strict."""
    # Create valid JSON first
    g = EIRGraph([], [])
    json_str = serialize_graph(g)
    data = json.loads(json_str)
    
    # Inject unknown factor
    data["factors"].append({
        "type": "UnknownMagicFactor",
        "scope": [],
        "weight": 1.0
    })
    bad_json = json.dumps(data)
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Unknown factor type"):
        deserialize_graph(bad_json, allow_unknown=False)

    # Should pass with warning if allow_unknown=True
    with pytest.warns(UserWarning):
        g2 = deserialize_graph(bad_json, allow_unknown=True)
        assert len(g2.factors) == 0

def test_strict_verify_unsupported_factor():
    """Test that VerifyCPSATBackend raises NotImplementedError on unsupported factor when strict."""
    class UnsupportedFactor(Factor):
        def eval(self, assign): return 0.0
        # No compile_cpsat method
        
    a = DiscreteVar("A", 2)
    f = UnsupportedFactor([a])
    g = EIRGraph([a], [f])
    
    # Strict = True -> Raise
    backend_strict = VerifyCPSATBackend(strict=True)
    with pytest.raises(NotImplementedError, match="does not support CP-SAT"):
        backend_strict.solve(g)
        
    # Strict = False -> Return result with unsupported info
    backend_lax = VerifyCPSATBackend(strict=False)
    res = backend_lax.solve(g)
    assert res.unsupported_factors is not None
    assert "UnsupportedFactor" in res.unsupported_factors

def test_implication_name_collision():
    """Test that multiple ImplicationFactors do not cause variable name collisions in CP-SAT."""
    # A->B, B->C
    a = DiscreteVar("A", 2)
    b = DiscreteVar("B", 2)
    c = DiscreteVar("C", 2)
    
    f1 = ImplicationFactor([a, b], state_a=1, state_b=1)
    f2 = ImplicationFactor([b, c], state_a=1, state_b=1)
    
    g = EIRGraph([a, b, c], [f1, f2])
    
    # Should solve without error
    backend = VerifyCPSATBackend(strict=True)
    res = backend.solve(g)
    assert res.status in ["OPTIMAL", "FEASIBLE"]
    
def test_soft_validation():
    """Test validation of soft assignments."""
    a = DiscreteVar("A", 3)
    g = EIRGraph([a], [])
    
    # Valid soft
    errs = g.validate({"A": [0.1, 0.8, 0.1]})
    assert len(errs) == 0
    
    # Invalid soft (sum)
    errs = g.validate({"A": [0.5, 0.5, 0.5]})
    assert len(errs) > 0
    assert "WARN" in errs[0]
    
    # Invalid soft (shape)
    errs = g.validate({"A": [0.5, 0.5]})
    assert len(errs) > 0
    assert "mismatch" in errs[0]

    # Negative
    errs = g.validate({"A": [-0.1, 1.1, 0.0]})
    assert len(errs) > 0
    assert "negative" in errs[0]
