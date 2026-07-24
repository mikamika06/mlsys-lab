import numpy as np


def _make_partials(logits, values, splits):
    ms = []
    ls = []
    os = []
    for a, b in splits:
        chunk_logits = logits[a:b]
        m = np.max(chunk_logits)
        w = np.exp(chunk_logits - m)
        ms.append(m)
        ls.append(np.sum(w))
        os.append(np.sum(w[:, None] * values[a:b], axis=0))
    return np.asarray(ms), np.asarray(ls), np.asarray(os)


def _oracle_full(logits, values):
    x = np.asarray(logits, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    m = np.max(x)
    w = np.exp(x - m)
    return np.sum(w[:, None] * v, axis=0) / np.sum(w)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    max_err = 0.0

    cases = [
        (37, 8, [0, 5, 13, 24, 37]),
        (101, 16, [0, 17, 40, 66, 101]),
        (23, 4, [0, 7, 12, 23]),
    ]

    for n, d, points in cases:
        logits = rng.normal(size=n).astype(np.float64) * 5.0
        values = rng.normal(size=(n, d)).astype(np.float64)
        splits = list(zip(points[:-1], points[1:]))

        ms, ls, os = _make_partials(logits, values, splits)
        ref = _oracle_full(logits, values)

        try:
            got = np.asarray(sol.merge_chunk_partials(ms, ls, os), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
