import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ..core import EIRGraph, DiscreteVar, ContinuousVar

@dataclass
class FieldResult:
    assignment: Dict[str, Any]
    energy: float
    iterations: int
    converged: bool

class FieldBackend:
    def __init__(self, steps=1000, lr=0.1, noise_start=0.1, noise_end=0.001, num_starts=1, 
                 grad_mode: str = "spsa", spsa_samples: int = 8):
        self.steps = steps
        self.lr = lr
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.num_starts = num_starts
        self.grad_mode = grad_mode
        self.spsa_samples = spsa_samples

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def solve(self, graph: EIRGraph, init_assign: Optional[Dict[str, Any]] = None) -> FieldResult:
        best_global_energy = float('inf')
        best_global_assign = None
        global_converged = False

        # 1. Layout parameters
        param_layout = [] 
        param_size = 0
        for v in graph.vars:
            if isinstance(v, ContinuousVar):
                start = param_size
                size = v.dim
                param_layout.append((v.name, 'cont', start, size))
                param_size += size
            elif isinstance(v, DiscreteVar):
                start = param_size
                size = v.n
                param_layout.append((v.name, 'disc', start, size))
                param_size += size

        # Helper to decode
        def decode(p, hard=False):
            assign = {}
            for name, vtype, start, size in param_layout:
                segment = p[start:start+size]
                if vtype == 'cont':
                    assign[name] = segment.copy()
                else:
                    if hard:
                        assign[name] = int(np.argmax(segment))
                    else:
                        assign[name] = self._softmax(segment)
            return assign

        def energy_fn(p):
            assign = decode(p, hard=False)
            return graph.energy(assign)

        # Multi-start loop
        for start_idx in range(self.num_starts):
            # Initialization
            params = np.random.randn(param_size) * 0.1
            
            # Warm start for first iteration
            if start_idx == 0 and init_assign is not None:
                for name, vtype, start, size in param_layout:
                    if name in init_assign:
                        val = init_assign[name]
                        if vtype == 'cont':
                            params[start:start+size] = np.array(val).flatten()
                        else:
                            # Expect logits or int
                            if isinstance(val, (int, np.integer)):
                                # Init logit for int
                                l = np.zeros(size)
                                l[val] = 5.0
                                params[start:start+size] = l
                            else:
                                params[start:start+size] = np.array(val)

            local_best_energy = float('inf')
            local_best_assign = None
            
            for i in range(self.steps):
                progress = i / self.steps
                noise_scale = self.noise_start * (1 - progress) + self.noise_end * progress
                
                # Gradient estimation
                grad = np.zeros_like(params)
                current_energy = energy_fn(params)
                epsilon = 1e-4
                
                if self.grad_mode == "finite_diff" and len(params) < 100:
                    for idx in range(len(params)):
                        old_val = params[idx]
                        params[idx] += epsilon
                        e_plus = energy_fn(params)
                        params[idx] = old_val
                        grad[idx] = (e_plus - current_energy) / epsilon
                else:
                    # SPSA
                    grad_accum = np.zeros_like(params)
                    for _ in range(self.spsa_samples):
                        delta = np.sign(np.random.randn(len(params)))
                        params_plus = params + epsilon * delta
                        params_minus = params - epsilon * delta
                        
                        e_plus = energy_fn(params_plus)
                        e_minus = energy_fn(params_minus)
                        
                        g_est = (e_plus - e_minus) / (2.0 * epsilon)
                        grad_accum += g_est * delta
                        
                    grad = grad_accum / self.spsa_samples

                params -= self.lr * grad
                if noise_scale > 0:
                    params += np.random.randn(len(params)) * noise_scale
                
                hard_assign = decode(params, hard=True)
                e_hard = graph.energy(hard_assign)
                
                if e_hard < local_best_energy:
                    local_best_energy = e_hard
                    local_best_assign = hard_assign
                
                if local_best_energy < 1e-6:
                    break
            
            # Update global best
            if local_best_energy < best_global_energy:
                best_global_energy = local_best_energy
                best_global_assign = local_best_assign
                
            if best_global_energy < 1e-6:
                break 
                
        return FieldResult(best_global_assign, best_global_energy, self.steps, best_global_energy < 1e-6)
