import numpy as np
from typing import List
from ..core import Factor, Variable, DiscreteVar

class OneHotFactor(Factor):
    """
    Penalizes deviations from a one-hot distribution.
    Mainly for the relaxation backend.
    Energy = sum(p * (1-p))  (Gini impurity-like) or Entropy.
    Or (sum(p) - 1)^2 + sum(p*(1-p))?
    
    Actually, for 'eval' on hard assignments, this is always 0 if valid.
    For soft assignments, it drives them to be spiky.
    """
    def __init__(self, scope: List[Variable], weight: float = 1.0):
        super().__init__(scope, weight)
        
    def eval(self, assign) -> float:
        # Usually applies to a single DiscreteVar
        v_name = self.scope[0].name
        val = assign[v_name]
        
        if isinstance(val, (int, np.integer)):
            return 0.0 # Hard assignment is always one-hot
            
        # Assume val is a probability vector or logits
        # If it's probability (sum~1), we want it to be close to 0 or 1.
        # Simple penalty: sum(x * (1-x)) for x in val
        p = np.array(val)
        # We also enforce sum(p) = 1 loosely here if the backend doesn't
        sum_err = (np.sum(p) - 1.0)**2
        sparsity = np.sum(p * (1.0 - p))
        # Return unweighted
        return sum_err + sparsity