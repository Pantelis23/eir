import numpy as np
from typing import List, Dict, Any, Union
from ..core import Factor, Variable, DiscreteVar, ContinuousVar

class DynamicsConsistencyFactor(Factor):
    """
    Penalizes ||z_{t+1} - f(z_t, a_t)||^2
    f is linear: z_{t+1} = A[a] @ z_t + b[a]
    f_table: dict mapping action_idx -> (A_matrix, b_vector)
    """
    def __init__(self, scope: List[Variable], f_table: Dict[int, Any], weight: float = 1.0):
        # scope: [z_t, z_t1, a_t]
        # z_t might be a fixed numpy array if t=0, but Factor scope expects Variables?
        # If z0 is fixed, we can handle it by not including it in scope but storing it?
        # Or easier: Factor scope must be variables. Fixed z0 is just a value in assign?
        # But 'assign' passed to eval only contains Variable values.
        # So we treat z0 as a constant property if it's not a variable.
        # But for general t, z_t is a variable.
        # Let's assume z_t and z_t1 are variables (or fixed inputs handled by caller via assignment).
        # Actually, if z0 is fixed, it's not in the graph vars. The factor can store it?
        # But factors are generic.
        # Let's assume the caller puts z0 into the assignment as a fixed value even if not optimized?
        # Or better: We treat z_t as scope[0]. If it's missing from assign, we fail.
        # The planner will ensure z0 is in the assignment.
        super().__init__(scope, weight)
        self.f_table = f_table
        # Pre-process table to numpy
        self._A = {}
        self._b = {}
        for k, (A, b) in f_table.items():
            self._A[k] = np.array(A)
            self._b[k] = np.array(b)
        self.n_actions = len(f_table)

    def eval(self, assign) -> float:
        # z_t: scope[0]
        # z_t1: scope[1]
        # a_t: scope[2]
        
        # Get z_t. It might be a fixed array (if not in graph vars but injected in assign)
        # or a variable value.
        z_t_name = self.scope[0].name if hasattr(self.scope[0], 'name') else "FIXED_Z0"
        # If scope[0] is just a name string? No, scope is list of Variables.
        # If z0 is fixed, the planner likely created a dummy Variable("z0") but didn't add it to graph vars list?
        # Yes, assign must contain it.
        z_t = np.array(assign[self.scope[0].name])
        
        z_t1 = np.array(assign[self.scope[1].name])
        a_val = assign[self.scope[2].name]
        
        # Compute expected A, b
        if isinstance(a_val, (int, np.integer)):
            # Hard assignment
            idx = int(a_val)
            A_eff = self._A[idx]
            b_eff = self._b[idx]
        else:
            # Soft assignment (probs)
            probs = np.array(a_val)
            probs = np.maximum(probs, 0.0)
            total = np.sum(probs)
            if total > 1e-9:
                probs /= total
            else:
                probs = np.ones_like(probs) / len(probs)
            
            # Expected A = sum p_k A_k
            # Assume all A same shape
            # Check shape of A[0]
            A_shape = self._A[0].shape
            b_shape = self._b[0].shape
            
            A_eff = np.zeros(A_shape)
            b_eff = np.zeros(b_shape)
            
            for k in range(self.n_actions):
                p = probs[k]
                if p > 1e-5:
                    A_eff += p * self._A[k]
                    b_eff += p * self._b[k]
                    
        # Predict
        z_pred = A_eff @ z_t + b_eff
        
        # L2 sq error
        diff = z_t1 - z_pred
        return np.sum(diff**2)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        # Serialize f_table
        tbl = {}
        for k in self.f_table:
            tbl[k] = (self._A[k].tolist(), self._b[k].tolist())
        d["f_table"] = tbl
        return d

class RewardFactor(Factor):
    """
    Returns cost = -beta * r(z_t, a_t)
    r(z, a) = w[a] . z + c[a]
    r_table: dict mapping action_idx -> (w_vector, c_scalar)
    """
    def __init__(self, scope: List[Variable], r_table: Dict[int, Any], beta: float = 1.0, weight: float = 1.0):
        # scope: [z_t, a_t]
        super().__init__(scope, weight)
        self.r_table = r_table
        self.beta = beta
        self._w = {}
        self._c = {}
        for k, (w, c) in r_table.items():
            self._w[k] = np.array(w)
            self._c[k] = float(c)
        self.n_actions = len(r_table)

    def eval(self, assign) -> float:
        z_t = np.array(assign[self.scope[0].name])
        a_val = assign[self.scope[1].name]
        
        # Compute expected w, c
        if isinstance(a_val, (int, np.integer)):
            idx = int(a_val)
            w_eff = self._w[idx]
            c_eff = self._c[idx]
        else:
            probs = np.array(a_val)
            probs = np.maximum(probs, 0.0)
            total = np.sum(probs)
            if total > 1e-9:
                probs /= total
            else:
                probs = np.ones_like(probs) / len(probs)
                
            w_eff = np.zeros_like(self._w[0])
            c_eff = 0.0
            for k in range(self.n_actions):
                p = probs[k]
                if p > 1e-5:
                    w_eff += p * self._w[k]
                    c_eff += p * self._c[k]
        
        r = np.dot(w_eff, z_t) + c_eff
        return -self.beta * r

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        tbl = {}
        for k in self.r_table:
            tbl[k] = (self._w[k].tolist(), self._c[k])
        d["r_table"] = tbl
        d["beta"] = self.beta
        return d
