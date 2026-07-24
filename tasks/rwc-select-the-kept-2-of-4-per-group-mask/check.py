import numpy as np

def _ref_mask(weights):
    groups = weights.reshape(-1, 4)
    abs_groups = np.abs(groups)
    idxs = np.argsort(-abs_groups, axis=1)[:, :2]
    row_indices = np.repeat(np.arange(groups.shape[0]), 2)
    col_indices = idxs.ravel()
    linear_indices = row_indices * 4 + col_indices
    mask = np.zeros_like(weights, dtype=bool)
    mask[linear_indices] = True
    return mask

def grade(sol, fx):
    cases = [
        np.array([0.5, -2.3, 1.1, 0.9,
                  -0.7, 3.2, -1.5, 0.4]),
        np.arange(8),
        np.random.default_rng(0).uniform(-10, 10, size=12),
    ]
    for w in cases:
        try:
            got = sol.select_top2_mask(w)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, np.ndarray) or got.dtype != bool:
            return {"exact_match": 0.0}
        ref = _ref_mask(w)
        if not np.array_equal(got, ref):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
