import numpy as np


def _oracle(q, chunks):
    q = np.asarray(q, dtype=np.float64)
    m = -np.inf
    l = 0.0
    out = np.zeros(chunks[0][1].shape[1], dtype=np.float64)

    for K, V in chunks:
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        scores = K @ q
        chunk_max = np.max(scores)
        new_m = max(m, chunk_max)

        old_scale = 0.0 if m == -np.inf else np.exp(m - new_m)
        weights = np.exp(scores - new_m)

        out = out * (l * old_scale) + weights @ V
        l = l * old_scale + np.sum(weights)
        m = new_m
        out = out / l

    return out


def grade(sol, fx) -> dict:
    cases = []
    rng = np.random.default_rng(7)
    for dims, sizes in [
        ((8, 5), [3, 4, 2]),
        ((16, 3), [1, 5, 7, 2]),
        ((32, 6), [4, 4, 4]),
    ]:
        d, h = dims
        q = rng.normal(size=d).astype(np.float32)
        chunks = []
        for n in sizes:
            K = rng.normal(size=(n, d)).astype(np.float32)
            V = rng.normal(size=(n, h)).astype(np.float32)
            chunks.append((K, V))
        cases.append((q, chunks))

    worst = 0.0
    for q, chunks in cases:
        try:
            got = np.asarray(sol.chunked_attention(q, chunks), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(q, chunks)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
