import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from planners.eir_planner import EIRActionSelector, apply_action_mask_like, EIR_AVAILABLE


def assert_eq(name, got, exp):
    ok = (got == exp) if np.isscalar(got) else np.array_equal(got, exp)
    if not ok:
        print(f"{name}: got={got} exp={exp} -> FAIL")
        raise SystemExit(1)
    # print(f"{name}: OK")


def main():
    # Use robust parameters for test
    sel = EIRActionSelector(steps=1000, num_starts=5, lr=0.1, noise_start=0.5)

    print("\n[Case 1] 1D + action_mask")
    logits = np.array([0.0, 10.0, 0.0, 5.0, 0.0], dtype=float)
    action_mask = np.array([1, 0, 1, 1, 1], dtype=float)
    a = sel.select(logits, action_dim=5, inventory_dim=0, action_mask=action_mask)
    assert_eq("case1", a, 3)

    print("\n[Case 2] 2D batch + action_mask + inventory")
    B, A = 3, 5
    logits2 = np.zeros((B, A), dtype=float)
    logits2[:, 1] = 10.0
    logits2[:, 3] = 5.0

    am2 = np.ones((B, A), dtype=float)
    am2[1, 1] = 0

    inv2 = np.ones((B, 2), dtype=float)
    inv2[2, 0] = 0

    acts = sel.select(logits2, action_dim=5, inventory_dim=2, action_mask=am2, inventory=inv2)
    assert_eq("case2", acts, np.array([1, 3, 1], dtype=np.int64))

    print("\n[Case 3] can_interact (B,2) any(axis=-1)")
    logits3 = np.zeros((2, 5), dtype=float)
    logits3[:, 0] = 20.0
    logits3[:, 1] = 5.0

    ci = np.array([[1, 1], [1, 0]], dtype=float)
    acts3 = sel.select(logits3, action_dim=5, inventory_dim=0, can_interact=ci, interact_action_idx=0)
    assert_eq("case3", acts3, np.array([0, 1], dtype=np.int64))

    print("\n[Case 4] inventory batch mismatch -> unchanged logits")
    logits4 = np.array([0.0, 10.0, 0.0, 0.0, 0.0], dtype=float)
    inv_mismatch = np.ones((2, 5), dtype=float)
    inv_mismatch[:, 1] = 0
    masked4 = apply_action_mask_like(logits4, inv_mismatch, None, action_dim=5, inventory_dim=5)
    a4 = int(np.argmax(masked4))
    assert_eq("case4", a4, 1)

    print("\n[Random Scenarios]")
    np.random.seed(42)
    n_random = 500
    for i in range(n_random):
        # Generate random logits and valid mask
        A_dim = np.random.randint(5, 20)
        logits_r = np.random.randn(A_dim)
        mask_r = (np.random.rand(A_dim) > 0.3).astype(int)
        
        # Ensure at least one valid
        if mask_r.sum() == 0: mask_r[0] = 1
        
        # Select
        try:
            choice = sel.select(logits_r, action_dim=A_dim, inventory_dim=0, action_mask=mask_r)
        except Exception as e:
            print(f"Random test {i} failed with exception: {e}")
            sys.exit(1)
            
        # Check Validity
        if mask_r[choice] == 0:
            print(f"Random test {i} FAIL: Selected invalid action {choice}")
            print(f"Mask: {mask_r}")
            sys.exit(1)
            
        # Check optimality logic
        masked_l = logits_r.copy()
        masked_l[mask_r == 0] = -1e9
        argmax_idx = np.argmax(masked_l)
        
        # Whether EIR available or not, for this class (discrete linear), 
        # it MUST match argmax (or tie).
        if choice != argmax_idx:
            # Check tie
            if not np.isclose(logits_r[choice], logits_r[argmax_idx]):
                 print(f"Random test {i} FAIL: Choice {choice} != Argmax {argmax_idx}")
                 sys.exit(1)

    print("SELECTOR: PASS")


if __name__ == "__main__":
    main()