from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Tuple, Optional, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

from .eir_plan_tape import EIRPlanTapePlanner

def systematic_resample(weights, rng):
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def particle_filter_step(z_particles, action, obs, obs_sigma, proc_sigma, f_true_table, rng):
    # Propagate
    A, b = f_true_table[action]
    A = np.array(A)
    b = np.array(b)
    
    # Vectorized update
    # z' = A @ z + b
    # (P, dim) -> (P, dim)
    z_next = (A @ z_particles[:, :, None]).squeeze(2) + b
    
    # Weight
    # w_i propto exp(-||o - z_i'||^2 / (2*obs_sigma^2))
    diff = z_next - obs
    dist_sq = np.sum(diff**2, axis=1)
    log_weights = -dist_sq / (2 * obs_sigma**2)
    
    # Stable softmax
    max_log = np.max(log_weights)
    weights = np.exp(log_weights - max_log)
    sum_w = np.sum(weights)
    if sum_w > 0:
        weights /= sum_w
    else:
        weights = np.ones(len(weights)) / len(weights)
        
    # Resample
    indices = systematic_resample(weights, rng)
    z_resampled = z_next[indices]
    
    # Process Noise
    noise = rng.normal(scale=proc_sigma, size=z_resampled.shape)
    z_final = z_resampled + noise
    
    return z_final

@dataclass
class EIRBeliefMPC:
    planner_params: Dict[str, Any]
    
    def run_episode(self, z0_true, z0_obs, f_true_table, f_model_table, r_table, constraints_fixed, 
                    T=10, H=5, P=64, obs_sigma=0.20, proc_sigma=0.03, seed=0, use_filter=True):
        
        rng = np.random.default_rng(seed)
        
        # Init Planner
        planner = EIRPlanTapePlanner(H=H, **self.planner_params)
        
        # Init Belief
        particles = z0_obs + 0.15 * rng.normal(size=(P, 2))
        
        z_true_curr = z0_true.copy()
        
        episode_log = {
            "actions": [],
            "z_true_traj": [z0_true],
            "total_cost": 0.0,
            "violations_fixed": 0,
            "violations_state": 0
        }
        
        total_cost = 0.0
        
        for t in range(T):
            # 1. Plan
            plan_res = planner.plan(
                z0_obs=particles, 
                f_true_table=f_true_table,
                f_model_table=f_model_table,
                r_table=r_table,
                constraints=constraints_fixed,
                sigma=0.0, 
                particles_eps=None, 
                num_particles=P,
                global_t0=t
            )
            
            # 2. Execute first action
            action = plan_res['actions'][0]
            episode_log['actions'].append(action)
            
            # Check constraints
            if t in constraints_fixed and action in constraints_fixed[t]:
                episode_log['violations_fixed'] += 1
                
            x, y = z_true_curr
            if y > 0.5 and action == 0: episode_log['violations_state'] += 1
            if y < -0.5 and action == 1: episode_log['violations_state'] += 1
            if abs(x) > 1.2 and action == 2: episode_log['violations_state'] += 1
            
            # Cost Calculation
            A_true, b_true = f_true_table[action]
            z_next_true = np.array(A_true) @ z_true_curr + np.array(b_true)
            
            A_mod, b_mod = f_model_table[action]
            z_pred_model = np.array(A_mod) @ z_true_curr + np.array(b_mod)
            
            consistency = np.sum((z_next_true - z_pred_model)**2)
            y_next = z_next_true[1]
            risk = max(0.0, abs(y_next) - self.planner_params['y_max'])**2
            
            w, c = r_table[action]
            reward = np.dot(w, z_true_curr) + c
            
            step_cost = self.planner_params['lambda_dyn'] * consistency + \
                        self.planner_params['lambda_risk'] * risk - \
                        self.planner_params['beta'] * reward
                        
            total_cost += step_cost
            
            # 3. Observe
            obs = z_next_true + obs_sigma * rng.normal(size=2)
            
            # 4. Filter or Propagate
            if use_filter:
                particles = particle_filter_step(
                    particles, action, obs, obs_sigma, proc_sigma, f_true_table, rng
                )
            else:
                # Baseline: just propagate dynamics + process noise (no reweight/resample)
                # Or just propagate mean? Benchmark says "particles fixed at initial" or "mean state".
                # "Static-belief MPC ... SKIP weighting/resample step. Only propagate particles with f_true and add proc noise".
                # OK.
                A, b = f_true_table[action]
                z_next = (np.array(A) @ particles[:, :, None]).squeeze(2) + np.array(b)
                noise = rng.normal(scale=proc_sigma, size=z_next.shape)
                particles = z_next + noise
            
            # Update state
            z_true_curr = z_next_true
            episode_log['z_true_traj'].append(z_true_curr)
            
        episode_log['total_cost'] = total_cost
        return episode_log
