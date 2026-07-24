import numpy as np


def _oracle_softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = [
        (rng.normal(size=(3, 17)), 4),
        (rng.normal(loc=50.0, scale=10.0, size=(2, 33)), 7),
        (rng.normal(loc=-100.0, scale=30.0, size=(4, 64)), 16),
        (np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]), 2),
    ]

    worst = 0.0
    for logits, chunk_size in cases:
        ref = _oracle_softmax(logits)
        try:
            got = sol.stream_softmax_row_chunks(logits.copy(), chunk_size)
            got = np.asarray(got, dtype=np.float64)
            err = float(np.max(np.abs(ref - got)))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
