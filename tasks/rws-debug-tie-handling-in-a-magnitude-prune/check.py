import numpy as np

def _oracle(weights, keep_fraction):
    n = len(weights)
    k = int(np.ceil(keep_fraction * n))
    if k <= 0:
        return np.zeros(n, dtype=bool)
    # stable sort by descending absolute value
    idx = np.argsort(-np.abs(weights), kind='mergesort')
    mask = np.zeros(n, dtype=bool)
    mask[idx[:k]] = True
    return mask

def grade(sol, fx) -> dict:
    cases = [
        (np.array([0.5, -1.2, 0.3, -1.2, 0.8]), 0.6),
        (np.array([0, 0, 0, 0]), 0.5),
        (np.arange(10), 0.7),
        (np.random.randn(15).astype(np.float32), 0.4),
        (np.array([-3, -3, -2, -1, 0]), 0.6),
    ]
    ok = 1.0
    for weights, keep in cases:
        try:
            got = sol.magnitude_prune_mask(weights, keep)
            if not isinstance(got, np.ndarray) or got.dtype != bool:
                ok = 0.0
                break
            expected = _oracle(weights, keep)
            if not np.array_equal(got, expected):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
