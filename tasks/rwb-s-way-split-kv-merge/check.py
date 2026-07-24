import numpy as np

from mlsys import scorers


def _dense_attention(q, k, v):
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    scores = (q @ k.T) * scale
    scores = scores - np.max(scores, axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=1, keepdims=True)
    return w @ v


def _local_partial(q, k_chunk, v_chunk):
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    scores = (q @ k_chunk.T) * scale
    m = np.max(scores, axis=1)
    p = np.exp(scores - m[:, None])
    l = np.sum(p, axis=1)
    acc = p @ v_chunk
    return m, l, acc


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []
    for n, d, S in [(10, 6, 1), (10, 6, 2), (10, 6, 3), (12, 8, 5), (30, 4, 7), (5, 4, 5)]:
        q = rng.normal(size=(n, d))
        k = rng.normal(size=(n, d))
        v = rng.normal(size=(n, d))
        scenarios.append((q, k, v, S))
    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for q, k, v, S in _scenarios():
        ref = _dense_attention(q, k, v)

        idx_chunks = np.array_split(np.arange(q.shape[0]), S)
        partials = [_local_partial(q, k[ix], v[ix]) for ix in idx_chunks]
        partials_arg = [(m.copy(), l.copy(), acc.copy()) for m, l, acc in partials]

        try:
            got = sol.merge_split_kv(partials_arg)
        except Exception:
            return {"max_abs_err": float("inf")}

        try:
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = scorers.max_abs_err(ref, got)
        if not np.isfinite(err):
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
