import sys

import numpy as np

BAD = {"max_abs_err": float("inf"), "state_bytes_ratio": float("inf")}


def _full_attention(Q, K, V, scale):
    scores = (Q @ K.T) * scale
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def _state_bytes(obj, seen):
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)

    if obj is None:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_state_bytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(_state_bytes(k, seen) + _state_bytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (bool, int, float, complex, np.number)):
        return 8
    if isinstance(obj, str):
        return len(obj)
    inner = getattr(obj, "__dict__", None)
    if inner is not None:
        return _state_bytes(inner, seen)
    return int(sys.getsizeof(obj))


def _run(sol, Q, K, V, num_blocks, scale):
    k_blocks = np.array_split(K, num_blocks)
    v_blocks = np.array_split(V, num_blocks)

    state = None
    peak = 0
    for kb, vb in zip(k_blocks, v_blocks):
        state = sol.ring_step(state, Q, kb.copy(), vb.copy(), scale)
        peak = max(peak, _state_bytes(state, set()))

    out = np.asarray(sol.ring_output(state), dtype=np.float64)
    return out, peak


def _cases():
    rng = np.random.default_rng(11)

    yield rng.normal(size=(64, 32)), rng.normal(size=(64, 32)), rng.normal(size=(64, 4)), 8
    yield rng.normal(size=(16, 24)), rng.normal(size=(16, 24)), rng.normal(size=(16, 2)), 1
    yield rng.normal(size=(10, 20)), rng.normal(size=(10, 20)), rng.normal(size=(10, 3)), 4
    big_q = rng.normal(size=(12, 16)) * 30.0
    big_k = rng.normal(size=(12, 16)) * 30.0
    yield big_q, big_k, rng.normal(size=(12, 2)), 6


def grade(sol, fx) -> dict:
    worst_err = 0.0
    worst_ratio = 0.0

    for Q, K, V, num_blocks in _cases():
        Q = np.ascontiguousarray(Q, dtype=np.float64)
        K = np.ascontiguousarray(K, dtype=np.float64)
        V = np.ascontiguousarray(V, dtype=np.float64)
        n, d = Q.shape
        dv = V.shape[1]
        scale = 1.0 / np.sqrt(d)

        ref = _full_attention(Q, K, V, scale)

        try:
            first, peak = _run(sol, Q, K, V, num_blocks, scale)
            second, _ = _run(sol, Q, K, V, num_blocks, scale)
        except Exception:
            return BAD

        if first.shape != ref.shape or second.shape != ref.shape:
            return BAD
        if not np.all(np.isfinite(first)):
            return BAD
        if not np.array_equal(first, second):
            return BAD

        budget = 8.0 * n * (dv + 2)
        worst_ratio = max(worst_ratio, peak / budget)
        worst_err = max(worst_err, float(np.max(np.abs(first - ref))))

    return {"max_abs_err": worst_err, "state_bytes_ratio": float(worst_ratio)}
