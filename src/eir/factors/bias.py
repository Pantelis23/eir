import numpy as np
from typing import List, Any, Dict
from ..core import Factor, Variable, DiscreteVar

class ActionBiasFactor(Factor):
    """
    Apply a static cost bias to each state of a DiscreteVar.
    
    Cost = sum_k p_k * costs[k]
    """
    def __init__(self, scope: List[Variable], costs: List[float], weight: float = 1.0, eps: float = 1e-12):
        super().__init__(scope, weight)
        if len(scope) != 1 or not isinstance(scope[0], DiscreteVar):
            raise ValueError("ActionBiasFactor requires exactly one DiscreteVar scope.")
        
        self.costs = np.array(costs, dtype=float)
        if len(self.costs) != scope[0].n:
            raise ValueError(f"Costs length {len(self.costs)} must match variable size {scope[0].n}")
        self.eps = eps

    def eval(self, assign) -> float:
        var_name = self.scope[0].name
        val = assign[var_name]
        
        if isinstance(val, (int, np.integer)):
            if 0 <= val < len(self.costs):
                return self.costs[val]
            return 0.0 # Should be caught by validation, but safe return
        
        # Soft assignment
        p = np.array(val)
        p = np.maximum(p, 0.0)
        total = np.sum(p)
        if total < self.eps:
            # Fallback to uniform distribution
            p = np.ones_like(p) / len(p)
        else:
            p = p / total
            
        return np.sum(p * self.costs)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["costs"] = self.costs.tolist()
        d["eps"] = self.eps
        return d
