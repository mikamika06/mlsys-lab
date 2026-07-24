import numpy as np


def _attention_reference(q, k, v, wo):
    b, h, s, d = q.shape
    heads = []
    scale = np.sqrt(float(d))
    for i in range(h):
        scores = np.matmul(q[:, i], np.transpose(k[:, i], (0, 2, 1))) / scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        heads.append(np.matmul(probs, v[:, i]))
    concat = np.concatenate(heads, axis=-1)
    return np.matmul(concat, wo)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (1, 4, 3, 2, 5, 2),
        (2, 6, 4, 3, 7, 3),
        (1, 8, 2, 4, 6, 4),
    ]

    best = 0.0
    for b, h, s, d, m, ranks in cases:
        q = rng.normal(size=(b, h, s, d)).astype(np.float64)
        k = rng.normal(size=(b, h, s, d)).astype(np.float64)
        v = rng.normal(size=(b, h, s, d)).astype(np.float64)
        wo = rng.normal(size=(h * d, m)).astype(np.float64)

        ref = _attention_reference(q, k, v, wo)

        try:
            got = sol.sharded_attention_heads(q, k, v, wo, ranks)
            err = float(np.max(np.abs(np.asarray(got) - ref)))
        except Exception:
            err = float("inf")

        if not np.isfinite(err):
            return {"max_abs_err": float("inf")}
        best = max(best, err)

    return {"max_abs_err": best}
