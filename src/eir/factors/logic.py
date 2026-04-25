import numpy as np
from typing import List, Dict, Any
from ..core import Factor, Variable, DiscreteVar

def _get_prob(val, target_idx):
    if isinstance(val, (int, np.integer)):
        return 1.0 if val == target_idx else 0.0
    # Assume soft vector
    return val[target_idx]

class ImplicationFactor(Factor):
    """
    A=state_a IMPLIES B=state_b
    Cost = P(A=state_a) * (1 - P(B=state_b))
    """
    def __init__(self, scope: List[Variable], state_a: int = 1, state_b: int = 1, weight: float = 1.0):
        super().__init__(scope, weight)
        self.state_a = state_a
        self.state_b = state_b

    def eval(self, assign) -> float:
        val_a = assign[self.scope[0].name]
        val_b = assign[self.scope[1].name]
        
        p_a = _get_prob(val_a, self.state_a)
        p_b = _get_prob(val_b, self.state_b)
        
        # Return unweighted
        return p_a * (1.0 - p_b)

    def compile_cpsat(self, model, varmap):
        a = varmap[self.scope[0].name]
        b = varmap[self.scope[1].name]
        
        # We need to enforce: (a == state_a) implies (b == state_b)
        # Reify (a == state_a) to a boolean variable.
        # Use id(self) to ensure uniqueness
        uid = id(self)
        is_a_active = model.NewBoolVar(f"{self.scope[0].name}_implies_trigger_{uid}")
        model.Add(a == self.state_a).OnlyEnforceIf(is_a_active)
        model.Add(a != self.state_a).OnlyEnforceIf(is_a_active.Not())
        
        # Enforce implication: if is_a_active, then b must be state_b
        model.Add(b == self.state_b).OnlyEnforceIf(is_a_active)
        
    def to_dict(self):
        d = super().to_dict()
        d.update({"state_a": self.state_a, "state_b": self.state_b})
        return d

class XorFactor(Factor):
    """
    A=1 XOR B=1
    (Assuming binary 0/1 for simplicity in logic, but checks specific states)
    Cost = (P(A) - P(B))^2 is not quite XOR. XOR means exactly one is true.
    Soft XOR: (P(A) + P(B) - 1)^2 (Constraint: Sum is 1)
    """
    def __init__(self, scope: List[Variable], target: int = 1, weight: float = 1.0):
        super().__init__(scope, weight)
        self.target = target # The 'true' state index

    def eval(self, assign) -> float:
        val_a = assign[self.scope[0].name]
        val_b = assign[self.scope[1].name]
        
        p_a = _get_prob(val_a, self.target)
        p_b = _get_prob(val_b, self.target)
        
        # Soft XOR: We want exactly one to be active.
        # (p_a + p_b - 1)^2 works well for "Exactly One" constraint between two vars.
        # Return unweighted
        return (p_a + p_b - 1.0)**2

    def compile_cpsat(self, model, varmap):
        a = varmap[self.scope[0].name]
        b = varmap[self.scope[1].name]
        
        uid = id(self)
        b_a = model.NewBoolVar(f"{self.scope[0].name}_is_{self.target}_{uid}_A")
        b_b = model.NewBoolVar(f"{self.scope[1].name}_is_{self.target}_{uid}_B")
        model.Add(a == self.target).OnlyEnforceIf(b_a)
        model.Add(a != self.target).OnlyEnforceIf(b_a.Not())
        model.Add(b == self.target).OnlyEnforceIf(b_b)
        model.Add(b != self.target).OnlyEnforceIf(b_b.Not())
        
        model.AddBoolXOr([b_a, b_b])

    def to_dict(self):
        d = super().to_dict()
        d.update({"target": self.target})
        return d

class AndFactor(Factor):
    """
    Enforces A=state_a AND B=state_b
    """
    def __init__(self, scope: List[Variable], state_a: int=1, state_b: int=1, weight: float=1.0):
        super().__init__(scope, weight)
        self.state_a = state_a
        self.state_b = state_b
        
    def eval(self, assign) -> float:
        val_a = assign[self.scope[0].name]
        val_b = assign[self.scope[1].name]
        p_a = _get_prob(val_a, self.state_a)
        p_b = _get_prob(val_b, self.state_b)
        
        # We want both to be 1.
        # Cost = (1-p_a) + (1-p_b)
        # Return unweighted
        return (1.0 - p_a) + (1.0 - p_b)

    def compile_cpsat(self, model, varmap):
        a = varmap[self.scope[0].name]
        b = varmap[self.scope[1].name]
        model.Add(a == self.state_a)
        model.Add(b == self.state_b)

    def to_dict(self):
        d = super().to_dict()
        d.update({"state_a": self.state_a, "state_b": self.state_b})
        return d