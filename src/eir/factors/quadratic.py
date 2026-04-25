import numpy as np
from typing import List, Any, Dict
from ..core import Factor, Variable, ContinuousVar

class QuadraticFactor(Factor):
    def __init__(self, scope: List[Variable], target: np.ndarray, weight: float = 1.0):
        super().__init__(scope, weight)
        self.target = np.array(target)

    def eval(self, assign) -> float:
        # Assumes scope[0] is a ContinuousVar
        val = assign[self.scope[0].name]
        diff = val - self.target
        # Return unweighted cost
        return np.sum(diff**2)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["target"] = self.target.tolist()
        return d