import numpy as np


def _score_matrix(q, k, dtype):
    n, d = q.shape
    m = k.shape[0]
    scores = np.empty((n, m), dtype=dtype)
    for i in range(n):
        for j in range(m):
            acc = dtype(0)
            for t in range(d):
                acc = dtype(acc + dtype(q[i, t] * k[j, t]))
            scores[i, j] = acc
    return scores


def _attention(q, k, v, dtype):
    d = q.shape[1]
    scores = _score_matrix(q, k, dtype).astype(np.float64)
    scores = scores / np.sqrt(d)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ v.astype(np.float64)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(3, 64)) * 2).astype(np.float32)
    k = (rng.normal(size=(5, 64)) * 2).astype(np.float32)
    v = rng.normal(size=(5, 8)).astype(np.float32)

    ref_fp16 = _attention(q, k, v, np.float16)
    ref_fp32 = _attention(q, k, v, np.float32)

    try:
        got = np.asarray(sol.attention_fp16_scores(q, k, v), dtype=np.float64)
    except Exception:
        return {
            "max_abs_err": float("inf"),
            "fp16_fp32_divergence": 0.0
        }

    return {
        "max_abs_err": float(np.max(np.abs(got - ref_fp16))),
        "fp16_fp32_divergence": float(np.max(np.abs(ref_fp16 - ref_fp32)))
    }
