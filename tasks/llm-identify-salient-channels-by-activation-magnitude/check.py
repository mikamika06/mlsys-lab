import numpy as np

def _reference_indices(X: np.ndarray, fraction: float) -> np.ndarray:
    n_channels = X.shape[1]
    k = int(np.ceil(fraction * n_channels))
    if k == 0:
        return np.array([], dtype=int)
    mean_abs = np.mean(np.abs(X), axis=0)
    # indices sorted descending by mean_abs
    idx_desc = np.argsort(-mean_abs)
    topk = idx_desc[:k]
    return np.sort(topk)

def grade(sol, fx) -> dict:
    # Prepare deterministic random data
    rng = np.random.default_rng(42)
    cases = [
        (np.array([[1, -2, 3], [4, -5, 6]]), 0.33),
        (rng.standard_normal((10, 5)), 0.5),
        (np.zeros((3, 7)), 0.2),
        (np.arange(12).reshape(4, 3), 1.0),
        (np.array([[0, 0], [0, 0]]), 0.0),
    ]
    ok = 1.0
    for X, frac in cases:
        try:
            got = sol.salient_channels(X, fraction=frac)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference_indices(X, frac)
        if not isinstance(got, np.ndarray) or got.dtype.kind != 'i':
            return {"exact_match": 0.0}
        if got.shape != ref.shape or not np.array_equal(got, ref):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
