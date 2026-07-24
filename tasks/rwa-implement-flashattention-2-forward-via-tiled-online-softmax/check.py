import tracemalloc

import numpy as np


def _naive_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V


def _peak_traced_bytes(fn, arg_factory, kwargs):
    """Peak bytes numpy/CPython actually allocate during one call to `fn`.

    A warm-up call (untracked) settles one-time lazy allocator setup so the
    tracked measurement is exactly reproducible across processes. Fresh
    argument copies are built for each call (outside the tracked region, so
    the harness's own copying never counts against the solution) so an
    in-place-mutating implementation can't corrupt the warm-up or measured
    inputs.
    """
    warm_args = arg_factory()
    fn(*warm_args, **kwargs)
    tracked_args = arg_factory()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = fn(*tracked_args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def _cases():
    rng = np.random.default_rng(2026)
    specs = [
        (128, 16, 16),
        (256, 16, 32),
        (384, 8, 32),
        (512, 8, 64),
    ]
    out = []
    for N, d, bs in specs:
        Q = rng.standard_normal((N, d)).astype(np.float64)
        K = rng.standard_normal((N, d)).astype(np.float64)
        V = rng.standard_normal((N, d)).astype(np.float64)
        out.append((Q, K, V, bs))
    return out


FAIL = {"max_abs_err": float("inf"), "peak_alloc_ratio": float("inf")}


def grade(sol, fx) -> dict:
    worst_err = 0.0
    worst_ratio = 0.0
    for Q, K, V, bs in _cases():
        ref = _naive_attention(Q, K, V)
        try:
            got, peak_bytes = _peak_traced_bytes(
                sol.flash_attention_forward,
                lambda Q=Q, K=K, V=V: (Q.copy(), K.copy(), V.copy()),
                {"block_size": bs},
            )
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return dict(FAIL)

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return dict(FAIL)

        worst_err = max(worst_err, float(np.max(np.abs(got - ref))))

        nxn_bytes = Q.shape[0] * Q.shape[0] * 8  # one float64 N x N matrix
        worst_ratio = max(worst_ratio, peak_bytes / nxn_bytes)

    return {"max_abs_err": worst_err, "peak_alloc_ratio": worst_ratio}
