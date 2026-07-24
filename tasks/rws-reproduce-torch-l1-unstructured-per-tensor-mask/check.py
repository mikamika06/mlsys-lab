import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    ok = 1.0
    for _ in range(10):
        n = rng.integers(5, 50)
        d = rng.integers(2, 10)
        weight = rng.standard_normal((n, d))
        amount = rng.random() * 0.9
        try:
            got_mask = sol.l1_unstructured_mask(weight, amount)
        except Exception:
            return {"exact_match": 0.0}
        if got_mask.shape != weight.shape or got_mask.dtype not in (np.bool_, np.int8, np.uint8):
            return {"exact_match": 0.0}
        flat = weight.ravel()
        abs_flat = np.abs(flat)
        k = int(np.floor(amount * flat.size))
        if k <= 0:
            ref_mask_flat = np.ones_like(flat, dtype=bool)
        else:
            sorted_idx = np.argsort(abs_flat)
            ref_mask_flat = np.ones_like(flat, dtype=bool)
            ref_mask_flat[sorted_idx[:k]] = False
        ref_mask = ref_mask_flat.reshape(weight.shape)
        if not np.array_equal(got_mask, ref_mask):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
