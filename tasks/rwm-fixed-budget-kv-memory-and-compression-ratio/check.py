import numpy as np

def _reference(keys, values, budget):
    T = len(keys)
    d = values.shape[1]
    norms = np.linalg.norm(values, axis=1)
    idx_sorted = np.argsort(norms)[::-1]  # descending order
    k = min(budget, T)
    selected_idx = idx_sorted[:k]
    selected_keys = keys[selected_idx]
    kept_bytes = (len(selected_keys) * d + len(selected_keys)) * values.dtype.itemsize
    full_bytes = (T * d + T) * values.dtype.itemsize
    ratio = full_bytes / kept_bytes if kept_bytes else float('inf')
    return set(selected_keys), ratio

def grade(sol, fx):
    cases = [
        # deterministic small case
        (np.array([1, 2, 3]), np.array([[1., 0.], [0., 2.], [3., 4.]]), 2),
        # random larger
        (np.arange(10), np.random.randn(10, 5).astype(np.float64), 4),
        # budget exceeds number of entries
        (np.arange(5), np.random.randn(5, 3).astype(np.float64), 10),
        # single element
        (np.array([42]), np.array([[7., -1.]]), 1),
    ]
    metrics = {"size_ratio_err": 0.0, "exact_keys": 0.0}
    for keys, values, budget in cases:
        try:
            out = sol.fixed_budget_kv(keys, values, budget)
        except Exception:
            return {"size_ratio_err": 1.0, "exact_keys": 0.0}
        if not isinstance(out, dict):
            return {"size_ratio_err": 1.0, "exact_keys": 0.0}
        # candidate ratio
        kept_bytes = (len(out) * values.shape[1] + len(out)) * values.dtype.itemsize
        full_bytes = (len(keys) * values.shape[1] + len(keys)) * values.dtype.itemsize
        cand_ratio = full_bytes / kept_bytes if kept_bytes else float('inf')
        # reference
        ref_keys, ref_ratio = _reference(keys, values, budget)
        exact = 1.0 if set(out.keys()) == ref_keys else 0.0
        rel_err_val = abs(cand_ratio - ref_ratio) / (abs(ref_ratio)+1e-12)
        metrics["size_ratio_err"] += rel_err_val
        metrics["exact_keys"] += exact
    n = len(cases)
    metrics["size_ratio_err"] /= n
    metrics["exact_keys"] /= n
    return metrics
