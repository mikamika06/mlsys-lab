import numpy as np


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def _reference(logits, temperature, seed):
    """Oracle: temperature softmax + per-step inverse-CDF draw from one rng.

    Consumes a single np.random.default_rng(seed), drawing one uniform per step
    in order. Computes the reference id sequence with NumPy — never hard-coded.
    """
    logits = np.asarray(logits, dtype=np.float64)
    T, V = logits.shape
    rng = np.random.default_rng(seed)
    ids = np.empty(T, dtype=np.int64)
    for t in range(T):
        p = _softmax(logits[t] / temperature)
        cdf = np.cumsum(p)
        u = rng.random()
        idx = int(np.searchsorted(cdf, u, side="right"))
        if idx >= V:
            idx = V - 1
        ids[t] = idx
    return ids


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    try:
        for _ in range(6):
            T = int(rng.integers(3, 12))
            V = int(rng.integers(4, 40))
            logits = rng.standard_normal((T, V)).astype(np.float64)
            temperature = float(rng.uniform(0.3, 1.8))
            seed = int(rng.integers(0, 1_000_000))

            got = sol.sample_sequence(logits, temperature, seed)
            ref = _reference(logits, temperature, seed)

            if not isinstance(got, np.ndarray):
                ok = 0.0
                break
            got = np.asarray(got)
            if got.shape != ref.shape or got.dtype != np.int64:
                ok = 0.0
                break
            if not np.array_equal(got, ref):
                ok = 0.0
                break
    except Exception:
        ok = 0.0

    return {"exact_match": ok}
