import numpy as np

def _oracle_mask(weights):
    if weights.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be a multiple of 4")
    abs_w = np.abs(weights)
    reshaped = abs_w.reshape(-1, 4)
    # indices of two largest per row
    idx = np.argpartition(reshaped, -2, axis=1)[:, -2:]
    mask_flat = np.zeros_like(reshaped, dtype=bool)
    rows = np.arange(mask_flat.shape[0])[:, None]
    mask_flat[rows, idx] = True
    return mask_flat.reshape(weights.shape)

def grade(sol, fx) -> dict:
    ok = 1.0
    rng = np.random.default_rng(42)
    for _ in range(5):
        n = rng.integers(2, 6)
        m = rng.choice([8, 12, 16])
        weights = rng.standard_normal((n, m)).astype(np.float32)
        try:
            got = sol.magnitude_optimal_2to4_mask(weights)
        except Exception:
            return {"exact_match": 0.0}
        if got.shape != weights.shape or got.dtype != np.bool_:
            return {"exact_match": 0.0}
        oracle = _oracle_mask(weights)
        if not np.array_equal(got, oracle):
            return {"exact_match": 0.0}
        # verify retained magnitude sum is maximal
        kept_sum = np.sum(np.abs(weights)[got])
        oracle_sum = np.sum(np.abs(weights)[oracle])
        if abs(kept_sum - oracle_sum) > 1e-9 * (oracle_sum + 1e-12):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
