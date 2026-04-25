from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Tuple, Optional, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Optional torch
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# Optional EIR
try:
    from eir.core import EIRGraph, DiscreteVar, ContinuousVar
    from eir.factors.bias import ActionBiasFactor
    from eir.factors.dynamics import DynamicsConsistencyFactor, RewardFactor
    from eir.factors.onehot import OneHotFactor
    from eir.backends.field import FieldBackend
    EIR_AVAILABLE = True
except Exception:
    EIR_AVAILABLE = False


@dataclass
class EIRPlanTapePlanner:
    H: int = 6
    z_dim: int = 2
    K: int = 6
    beta: float = 1.0
    lambda_dyn: float = 2.0
    lambda_risk: float = 5.0
    y_max: float = 0.8
    steps: int = 5
    grad_mode: str = "spsa"
    beam_width: int = 32
    local_search_candidates: int = 10
    local_search_passes: int = 2
    num_seeds: int = 6
    
    # New params
    eps_forbid: float = 0.05
    cvar_q: float = 0.10
    lambda_cvar: float = 0.5
    
    def plan(self, z0_obs: Any, f_true_table: Dict, f_model_table: Dict, r_table: Dict, constraints: Optional[Dict[int, List[int]]] = None, sigma: float = 0.15, particles_eps: Optional[np.ndarray] = None, num_particles: int = 32) -> Dict[str, Any]:
        """
        z0_obs: observed initial state
        particles_eps: (P, z_dim) standard normal noise. If None, generated.
        """
        if not EIR_AVAILABLE:
            raise RuntimeError("EIR not installed")
            
        z0_obs = np.array(z0_obs)
        constraints = constraints or {}
        
        if particles_eps is None:
            particles_eps = np.random.randn(num_particles, self.z_dim)
            
        # Initial particle set: (P, z_dim)
        z0_particles = z0_obs + sigma * particles_eps
        P = len(z0_particles)
        
        # Build Graph (same as before, mostly for refinement phase)
        # Note: Refinement usually works on mean/mode? Or single particle?
        # Benchmark says "Use only top M=6 as seeds for FieldBackend refinement".
        # Refinement with FieldBackend optimizes deterministic problem from z0_obs?
        # Or z0_mean?
        # Let's assume refinement uses z0_obs (mean) to produce a candidate sequence, 
        # but evaluation (ranking) uses particles.
        # This keeps backend standard.
        
        a_vars = [DiscreteVar(f"a{t}", self.K) for t in range(self.H)]
        z_vars = [ContinuousVar(f"z{t}", self.z_dim) for t in range(self.H + 1)] 
        
        factors = []
        from eir.factors.quadratic import QuadraticFactor
        factors.append(QuadraticFactor([z_vars[0]], target=z0_obs, weight=1e6))
        
        for t in range(self.H):
            dyn = DynamicsConsistencyFactor([z_vars[t], z_vars[t+1], a_vars[t]], f_model_table, weight=self.lambda_dyn)
            factors.append(dyn)
            rew = RewardFactor([z_vars[t], a_vars[t]], r_table, beta=self.beta, weight=1.0)
            factors.append(rew)
            if t in constraints:
                forbidden = constraints[t]
                costs = np.zeros(self.K)
                for idx in forbidden:
                    costs[idx] = 1e9
                bias = ActionBiasFactor([a_vars[t]], costs=costs, weight=1.0)
                factors.append(bias)
            factors.append(OneHotFactor([a_vars[t]], weight=0.1))

        graph = EIRGraph(vars=z_vars + a_vars, factors=factors)
        
        # Search Strategy: Beam Search + Local Improvement
        self.eval_count = 0
        
        # Precompute arrays for speed
        F_A = np.zeros((self.K, self.z_dim, self.z_dim))
        F_b = np.zeros((self.K, self.z_dim))
        R_w = np.zeros((self.K, self.z_dim))
        R_c = np.zeros(self.K)
        M_A = np.zeros((self.K, self.z_dim, self.z_dim))
        M_b = np.zeros((self.K, self.z_dim))
        
        for k in range(self.K):
            F_A[k], F_b[k] = f_true_table[k]
            R_w[k], R_c[k] = r_table[k]
            M_A[k], M_b[k] = f_model_table[k]

        def compute_cvar(values, q):
            # values: (P,) array
            # q: fraction (e.g. 0.10)
            # Take mean of top ceil(q*P)
            if q <= 0: return 0.0
            if q >= 1.0: return np.mean(values)
            
            k = int(np.ceil(q * len(values)))
            if k == 0: k = 1 # Safety
            
            # Sort descending or partition
            # partition is faster if P is large, sort is fine for 32
            # Use argpartition to get top k
            top_k_indices = np.argpartition(values, -k)[-k:]
            top_vals = values[top_k_indices]
            return np.mean(top_vals)

        def get_forbidden_fraction(t, z_batch):
            # z_batch: (P, z_dim)
            # Returns array (K,) of forbidden fraction for each action
            fractions = np.zeros(self.K)
            fixed_forbidden = set(constraints.get(t, []))
            
            x = z_batch[:, 0]
            y = z_batch[:, 1]
            
            # Constraints logic per particle
            # 0 forbidden if y > 0.5
            # 1 forbidden if y < -0.5
            # 2 forbidden if |x| > 1.2
            
            mask_0 = (y > 0.5)
            mask_1 = (y < -0.5)
            mask_2 = (np.abs(x) > 1.2)
            
            P = len(z_batch)
            fractions[0] = np.sum(mask_0) / P
            fractions[1] = np.sum(mask_1) / P
            fractions[2] = np.sum(mask_2) / P
            
            # Fixed constraints apply to ALL particles (fraction 1.0)
            for f in fixed_forbidden:
                fractions[f] = 1.0
                
            return fractions

        def evaluate_seq(seq, z_init_particles):
            self.eval_count += 1
            z_curr = z_init_particles.copy() # (P, z_dim)
            
            # We track mean trajectory for result
            traj = [np.mean(z_curr, axis=0)]
            
            energies_p = np.zeros(len(z_curr))
            
            for t, a in enumerate(seq):
                # Check chance constraints
                fracs = get_forbidden_fraction(t, z_curr)
                if fracs[a] > self.eps_forbid:
                    return float('inf'), []

                # Vectorized over particles
                # Reward (P,)
                # r = w . z + c
                r_p = np.sum(R_w[a] * z_curr, axis=1) + R_c[a]
                
                # Consistency
                z_pred = (M_A[a] @ z_curr[:, :, None]).squeeze(2) + M_b[a] # (P, z)
                z_next = (F_A[a] @ z_curr[:, :, None]).squeeze(2) + F_b[a] # (P, z)
                
                diff = z_next - z_pred
                c_cost_p = np.sum(diff**2, axis=1)
                
                # Risk
                y_next = z_next[:, 1]
                risk_p = np.maximum(0.0, np.abs(y_next) - self.y_max)**2
                
                step_E_p = self.lambda_dyn * c_cost_p + self.lambda_risk * risk_p - self.beta * r_p
                energies_p += step_E_p
                
                z_curr = z_next
                traj.append(np.mean(z_curr, axis=0))
                
            mean_E = np.mean(energies_p)
            cvar_E = compute_cvar(energies_p, self.cvar_q)
            J = mean_E + self.lambda_cvar * cvar_E
            
            return J, traj

        # 1. Beam Search
        # Beam items: (J_score, seq, z_particles, E_p_accum)
        beam = [(0.0, [], z0_particles, np.zeros(len(z0_particles)))]
        
        for t in range(self.H):
            new_candidates = []
            
            for score_acc, seq, z_curr, Ep_curr in beam:
                # Determine allowed actions
                fracs = get_forbidden_fraction(t, z_curr)
                allowed_actions = [k for k in range(self.K) if fracs[k] <= self.eps_forbid]
                
                for k in allowed_actions:
                    self.eval_count += 1
                    
                    # Reward (P,)
                    r_p = np.sum(R_w[k] * z_curr, axis=1) + R_c[k]
                    
                    # Update
                    z_pred = (M_A[k] @ z_curr[:, :, None]).squeeze(2) + M_b[k]
                    z_next = (F_A[k] @ z_curr[:, :, None]).squeeze(2) + F_b[k]
                    
                    diff = z_next - z_pred
                    c_cost_p = np.sum(diff**2, axis=1)
                    
                    y_next = z_next[:, 1]
                    risk_p = np.maximum(0.0, np.abs(y_next) - self.y_max)**2
                    
                    step_E_p = self.lambda_dyn * c_cost_p + self.lambda_risk * risk_p - self.beta * r_p
                    
                    new_Ep = Ep_curr + step_E_p
                    
                    # Compute J
                    mean_E = np.mean(new_Ep)
                    cvar_E = compute_cvar(new_Ep, self.cvar_q)
                    new_J = mean_E + self.lambda_cvar * cvar_E
                    
                    new_candidates.append((new_J, seq + [k], z_next, new_Ep))
            
            if not new_candidates:
                beam = []
                break
            
            new_candidates.sort(key=lambda x: x[0])
            beam = new_candidates[:self.beam_width]
            
        # 2. Local Improvement
        candidates = []
        for J, seq, _, _ in beam:
            candidates.append((J, seq))
            
        top_candidates = candidates[:self.local_search_candidates]
        final_candidates = []
        
        for init_J, init_seq in top_candidates:
            curr_seq = list(init_seq)
            curr_J = init_J
            
            for _ in range(self.local_search_passes):
                improved = False
                for t in range(self.H):
                    best_a = curr_seq[t]
                    best_val = curr_J
                    
                    for k in range(self.K):
                        if k == curr_seq[t]: continue
                        
                        test_seq = curr_seq[:]
                        test_seq[t] = k
                        
                        # Full Eval
                        J_val, _ = evaluate_seq(test_seq, z0_particles)
                        
                        if J_val < best_val:
                            best_val = J_val
                            best_a = k
                            improved = True
                    
                    curr_seq[t] = best_a
                    curr_J = best_val
                
                if not improved: break
            
            J_final, traj = evaluate_seq(curr_seq, z0_particles)
            if J_final != float('inf'):
                final_candidates.append((J_final, curr_seq, traj))
            
        final_candidates.sort(key=lambda x: x[0])
        best_seeds = final_candidates[:self.num_seeds]
        
        best_res_J = float('inf')
        best_res_assignment = None
        best_res_seq = None
        best_res_traj = None
        
        # Skip FieldBackend refinement because it optimizes single-path objective
        # (doesn't know about particles or CVaR).
        # Returning best seed result directly is robust.
        
        if best_seeds:
            best_res_J = best_seeds[0][0]
            best_res_seq = best_seeds[0][1]
            best_res_traj = best_seeds[0][2]
        else:
            raise RuntimeError("No valid plan found")

        final_actions = list(best_res_seq)
        # z_traj is mean trajectory from particles
        final_z = list(best_res_traj[1:]) # Skip z0
            
        return {
            "actions": final_actions,
            "z_traj": final_z,
            "energy": best_res_J,
            "eval_count": self.eval_count
        }