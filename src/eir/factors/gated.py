import numpy as np
from typing import List, Any, Dict, Union
from ..core import Factor, Variable, DiscreteVar

class GatedTargetsFactor(Factor):
    """
    Mixed discrete-continuous factor.
    
    Cost = sum_k (p_k * ||P_k - target_k||^2) + lam * sum_k ||P_k||^2
    
    Where:
      p_k is probability of action k (from Action variable)
      P_k is the k-th continuous parameter vector
      target_k is the target for P_k
      lam is regularization strength to keep unused parameters bounded.
    """
    def __init__(self, 
                 scope: List[Variable], 
                 targets: List[Any], 
                 lam: float = 1e-3, 
                 weight: float = 1.0,
                 eps: float = 1e-12):
        """
        scope: [ActionVar, ParamVar_0, ..., ParamVar_K-1]
        targets: list of K targets (arrays or floats)
        """
        super().__init__(scope, weight)
        self.targets = [np.array(t) for t in targets]
        self.lam = lam
        self.eps = eps
        
        # Validation
        action_var = scope[0]
        if not isinstance(action_var, DiscreteVar):
            raise ValueError("First variable in scope must be DiscreteVar (Action)")
        if len(scope) - 1 != len(self.targets):
            raise ValueError(f"Number of parameter vars ({len(scope)-1}) must match targets ({len(self.targets)})")

    def eval(self, assign) -> float:
        # 1. Get Action Probabilities
        action_name = self.scope[0].name
        val_action = assign[action_name]
        n_actions = self.scope[0].n
        
        probs = np.zeros(n_actions)
        if isinstance(val_action, (int, np.integer)):
            if 0 <= val_action < n_actions:
                probs[val_action] = 1.0
        else:
            # Soft vector
            raw_p = np.array(val_action)
            # Clamp and Normalize
            raw_p = np.maximum(raw_p, 0.0)
            total = np.sum(raw_p)
            if total < self.eps:
                probs[:] = 1.0 / n_actions # Fallback to uniform
            else:
                probs = raw_p / total

        cost_gated = 0.0
        cost_reg = 0.0
        
        # 2. Iterate parameters
        for k in range(n_actions):
            # Param k corresponds to scope[k+1]
            param_name = self.scope[k+1].name
            param_val = np.array(assign[param_name]) # Ensure array
            
            # Gated Cost
            p_k = probs[k]
            # Only compute distance if p_k is significant (optimization)
            if p_k > self.eps:
                diff = param_val - self.targets[k]
                cost_gated += p_k * np.sum(diff**2)
            
            # Regularization (always applied)
            cost_reg += np.sum(param_val**2)
            
        return cost_gated + (self.lam * cost_reg)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["targets"] = [t.tolist() for t in self.targets]
        d["lam"] = self.lam
        d["eps"] = self.eps
        return d
