import numpy as np

def _reference_indices(X: list[list[float]], fraction: float) -> list[int]:
    arr = np.array(X, dtype=float)
    n_channels = arr.shape[1]
    k = int(np.ceil(fraction * n_channels))
    if k == 0 or n_channels == 0:
        return []
    mean_abs = np.mean(np.abs(arr), axis=0)
    idx_desc = np.argsort(-mean_abs)
    topk = idx_desc[:k]
    return sorted(topk.tolist())

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        ([[1, -2, 3], [4, -5, 6]], 0.33),
        (rng.standard_normal((10, 5)).tolist(), 0.5),
        ([[0.0] * 7 for _ in range(3)], 0.2),
        (np.arange(12).reshape(4, 3).tolist(), 1.0),
        ([[0, 0], [0, 0]], 0.0),
    ]
    ok = 1.0
    for X, frac in cases:
        try:
            got = sol.salient_channels(X, fraction=frac)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference_indices(X, frac)
        if not isinstance(got, list) or not all(isinstance(i, int) for i in got):
            return {"exact_match": 0.0}
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
