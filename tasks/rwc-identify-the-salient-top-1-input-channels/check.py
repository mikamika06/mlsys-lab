import numpy as np

def _reference(X):
    axes = tuple(i for i in range(X.ndim) if i != 1)
    s = np.mean(np.abs(X), axis=axes)
    k = int(np.ceil(s.size * 0.01))
    top = np.argsort(-s)[:k]
    return sorted(int(idx) for idx in top)

def grade(sol, fx):
    rng = np.random.default_rng(12345)
    X = rng.standard_normal((4, 200, 3, 3))
    try:
        got = sol.identify_salient_channels(X)
    except Exception:
        return {"exact_match": 0.0}
    if not isinstance(got, (list, tuple)):
        return {"exact_match": 0.0}
    try:
        got_set = sorted(int(i) for i in got)
    except Exception:
        return {"exact_match": 0.0}
    ref = _reference(X)
    ok = 1.0 if got_set == ref else 0.0
    return {"exact_match": ok}
