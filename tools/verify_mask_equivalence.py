import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from planners.eir_planner import apply_action_mask_like

def reference_masking(logits, inventory, action_mask, action_dim, inventory_dim, 
                      can_interact, interact_idx, can_reproduce, reproduce_idx):
    """
    Reference implementation: slow, explicit loops, exact semantics.
    """
    # Semantics:
    # 1. All None -> Return Unchanged
    if inventory is None and action_mask is None and can_interact is None and can_reproduce is None:
        return np.array(logits)

    # 2. place_offset check -> Return Unchanged
    place_offset = action_dim - inventory_dim
    if place_offset < 0:
        return np.array(logits)

    # Convert to numpy 2D (batch, dim)
    l_np = np.array(logits)
    orig_ndim = l_np.ndim
    if orig_ndim == 1:
        working = l_np[None, :].copy()
    else:
        working = l_np.copy()
    
    B, A = working.shape

    # --- Inventory ---
    if inventory is not None:
        inv = np.array(inventory)
        if inv.ndim == 1: inv = inv[None, :]
        
        # Batch mismatch -> EARLY RETURN
        if inv.shape[0] != B:
            return working[0] if orig_ndim == 1 else working
        
        # Apply mask
        if inventory_dim > 0:
            for b in range(B):
                for i in range(inventory_dim):
                    # Check column bounds in inv and logits
                    if i < inv.shape[1]:
                        if inv[b, i] <= 0:
                            target_col = place_offset + i
                            if target_col < A:
                                working[b, target_col] = -1e9

    # --- Action Mask ---
    if action_mask is not None:
        am = np.array(action_mask)
        if am.ndim == 1: am = am[None, :]
        
        # Apply only if shapes align
        if am.shape[0] == B and am.shape[1] == A:
            for b in range(B):
                for i in range(A):
                    if am[b, i] <= 0:
                        working[b, i] = -1e9

    # --- Can Interact ---
    if can_interact is not None and interact_idx is not None:
        ci = np.array(can_interact)
        if ci.ndim == 0: ci = ci.reshape(1, 1)
        elif ci.ndim == 1: ci = ci[None, :]
        
        if ci.shape[0] == B and interact_idx < A:
            for b in range(B):
                # Invalid if ANY value in row <= 0
                is_invalid = False
                for k in range(ci.shape[1]):
                    if ci[b, k] <= 0:
                        is_invalid = True
                        break
                if is_invalid:
                    working[b, interact_idx] = -1e9

    # --- Can Reproduce ---
    if can_reproduce is not None and reproduce_idx is not None:
        cr = np.array(can_reproduce)
        if cr.ndim == 0: cr = cr.reshape(1, 1)
        elif cr.ndim == 1: cr = cr[None, :]
        
        if cr.shape[0] == B and reproduce_idx < A:
            for b in range(B):
                is_invalid = False
                for k in range(cr.shape[1]):
                    if cr[b, k] <= 0:
                        is_invalid = True
                        break
                if is_invalid:
                    working[b, reproduce_idx] = -1e9

    if orig_ndim == 1:
        return working[0]
    return working


def main():
    np.random.seed(0)
    print("Starting Mask Equivalence Verification...")
    
    n_tests = 500
    failures = 0
    
    for i in range(n_tests):
        # Randomized parameters
        is_1d = np.random.rand() < 0.3
        B = 1 if is_1d else np.random.randint(1, 10)
        A = np.random.randint(5, 20)
        
        logits = np.random.randn(A) if is_1d else np.random.randn(B, A)
        
        # Inventory
        inv_dim = np.random.randint(0, A + 2)
        inv = None
        if np.random.rand() < 0.7:
            # Maybe mismatch batch
            inv_B = B if np.random.rand() < 0.9 else B + 1
            inv = (np.random.randn(inv_B, inv_dim) > 0).astype(int)
            if is_1d and inv_B == 1 and np.random.rand() < 0.5:
                inv = inv[0] # Make it 1D

        # Action Mask
        am = None
        if np.random.rand() < 0.7:
            am_B = B if np.random.rand() < 0.9 else B + 1
            am_A = A if np.random.rand() < 0.9 else A + 1
            am = (np.random.randn(am_B, am_A) > 0).astype(int)
            if is_1d and am_B == 1 and am_A == A and np.random.rand() < 0.5:
                am = am[0]

        # Can Interact
        ci = None
        int_idx = None
        if np.random.rand() < 0.7:
            int_idx = np.random.randint(0, A)
            ci_B = B if np.random.rand() < 0.9 else B + 1
            ci_dim = np.random.randint(1, 4)
            ci = (np.random.randn(ci_B, ci_dim) > 0).astype(int)
            if is_1d and ci_B == 1 and ci_dim == 1 and np.random.rand() < 0.5:
                ci = ci[0,0] # scalar
            elif is_1d and ci_B == 1 and np.random.rand() < 0.5:
                ci = ci[0] # 1D

        # Can Reproduce
        cr = None
        rep_idx = None
        if np.random.rand() < 0.7:
            rep_idx = np.random.randint(0, A)
            cr_B = B if np.random.rand() < 0.9 else B + 1
            cr_dim = np.random.randint(1, 4)
            cr = (np.random.randn(cr_B, cr_dim) > 0).astype(int)
            if is_1d and cr_B == 1 and np.random.rand() < 0.5:
                cr = cr[0]

        # Run both
        ref = reference_masking(logits, inv, am, A, inv_dim, ci, int_idx, cr, rep_idx)
        got = apply_action_mask_like(logits, inv, am, A, inv_dim, ci, int_idx, cr, rep_idx)
        
        # Compare
        if not np.array_equal(ref, got):
            print(f"FAIL test {i}")
            print(f"Logits shape: {logits.shape}")
            if inv is not None: print(f"Inv shape: {inv.shape}")
            if am is not None: print(f"AM shape: {am.shape}")
            print(f"Ref:\n{ref}")
            print(f"Got:\n{got}")
            failures += 1
            if failures >= 5: break

    if failures == 0:
        print("MASK EQUIVALENCE: PASS")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
