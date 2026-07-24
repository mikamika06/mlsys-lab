import numpy as np

def grade(sol, fx) -> dict:
    # Reference implementation using NumPy
    def reference(logits, k):
        sorted_idx = np.argsort(-logits)
        mask = np.zeros_like(logits, dtype=bool)
        mask[sorted_idx[:k]] = True
        filtered = logits.copy()
        filtered[~mask] = -np.inf
        return mask, filtered

    # Test cases: diverse shapes and tie situations
    cases = [
        (np.array([0.5, 2.3, 1.7]), 1),
        (np.array([-1.0, -0.5, -2.0, -0.5]), 2),   # ties at -0.5
        (np.arange(10).astype(float), 5),
        (np.random.randn(20), 3)
    ]

    for logits, k in cases:
        try:
            mask, filtered = sol.top_k_filter(logits.copy(), k)
        except Exception:
            return {"exact_match": 0.0}

        ref_mask, ref_filtered = reference(logits, k)

        if not (mask.shape == ref_mask.shape and filtered.shape == ref_filtered.shape):
            return {"exact_match": 0.0}
        if not np.array_equal(mask, ref_mask):
            return {"exact_match": 0.0}
        if not np.allclose(filtered, ref_filtered, atol=1e-12, rtol=0):
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
