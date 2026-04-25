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
    from eir.factors.gated import GatedTargetsFactor
    from eir.backends.field import FieldBackend
    EIR_AVAILABLE = True
except Exception:
    EIR_AVAILABLE = False


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class EIRMixedPlanner:
    logit_temp: float = 1.0
    invalid_penalty: float = 1e6
    reg_lambda: float = 0.01

    steps: int = 10
    lr: float = 0.1
    noise_start: float = 0.0
    noise_end: float = 0.0
    num_starts: int = 1

    def select(
        self,
        logits: Any,
        targets: Any, 
        *,
        action_dim: int,
        param_dim: int,
        inventory_dim: int = 0,
        inventory: Any | None = None,
        action_mask: Any | None = None,
    ) -> Tuple[Any, Any]:
        from planners.eir_planner import apply_action_mask_like, masked_argmax

        is_torch = TORCH_AVAILABLE and isinstance(logits, torch.Tensor)
        device = logits.device if is_torch else None

        np_logits = _to_numpy(logits)
        np_targets = _to_numpy(targets)
        
        masked = apply_action_mask_like(
            action_logits=logits,
            inventory=inventory,
            action_mask=action_mask,
            action_dim=action_dim,
            inventory_dim=inventory_dim,
        )
        
        if masked.ndim == 1:
            masked = masked[None, :]
            np_logits = np_logits[None, :]
            np_targets = np_targets[None, :]
            
        B, A = masked.shape
        
        if np_targets.ndim == 2: 
             np_targets = np.tile(np_targets[None, :, :], (B, 1, 1))
             
        actions = np.zeros(B, dtype=np.int64)
        chosen_params = np.zeros((B, param_dim), dtype=np.float64)
        
        for i in range(B):
            row_masked = masked[i]
            row_raw = np_logits[i]
            row_targets = np_targets[i]
            
            valid = row_masked > -1e8
            if not valid.any():
                actions[i] = 0
                continue
                
            best_idx = masked_argmax(row_masked)
            best_param = row_targets[best_idx]

            if EIR_AVAILABLE:
                try:
                    l = row_raw / max(self.logit_temp, 1e-8)
                    max_valid = float(np.max(l[valid])) if valid.any() else 0.0
                    l_shift = l - max_valid
                    costs = (-l_shift) + (self.invalid_penalty * (~valid).astype(np.float64))
                    
                    action_var = DiscreteVar("Action", A)
                    
                    param_vars = []
                    target_list = []
                    init_assign = {}
                    
                    for k in range(A):
                        p_var = ContinuousVar(f"P{k}", param_dim)
                        param_vars.append(p_var)
                        target_list.append(row_targets[k])
                        init_assign[f"P{k}"] = row_targets[k]
                        
                    # Heuristic Init for Logits
                    target_norms_sq = np.sum(row_targets**2, axis=1)
                    energies = -row_raw + self.reg_lambda * target_norms_sq
                    energies[~valid] = 1e9
                    
                    best_heuristic_idx = np.argmin(energies)
                    logits_init = np.zeros(A)
                    logits_init[best_heuristic_idx] = 5.0
                    init_assign["Action"] = logits_init
                    
                    gated = GatedTargetsFactor(
                        [action_var] + param_vars,
                        targets=target_list,
                        lam=self.reg_lambda,
                        weight=1.0
                    )
                    
                    bias = ActionBiasFactor([action_var], costs=costs.tolist(), weight=1.0)
                    
                    graph = EIRGraph(
                        vars=[action_var] + param_vars,
                        factors=[gated, bias]
                    )
                    
                    backend = FieldBackend(
                        steps=self.steps,
                        lr=self.lr,
                        noise_start=self.noise_start,
                        noise_end=self.noise_end,
                        num_starts=self.num_starts,
                        grad_mode="spsa",
                        spsa_samples=1
                    )
                    
                    res = backend.solve(graph, init_assign=init_assign)
                    choice = int(res.assignment["Action"])
                    
                    if 0 <= choice < A and valid[choice]:
                        best_idx = choice
                        p_key = f"P{choice}"
                        if p_key in res.assignment:
                            best_param = res.assignment[p_key]
                        else:
                            best_param = row_targets[choice]
                            
                except Exception as e:
                    logger.debug(f"EIR Mixed solve failed: {e}")
                    raise e
                    
            actions[i] = best_idx
            chosen_params[i] = best_param

        if is_torch:
            return (
                torch.tensor(actions, device=device, dtype=torch.long),
                torch.tensor(chosen_params, device=device, dtype=logits.dtype)
            )
        
        return actions, chosen_params