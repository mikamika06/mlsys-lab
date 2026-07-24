import numpy as np

from mlsys import scorers


def _ref_gqa_attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    H_q, n, d = q.shape
    H_kv = k.shape[0]
    n_rep = H_q // H_kv

    k_rep = np.repeat(k, n_rep, axis=0)  # (H_q, n, d)
    v_rep = np.repeat(v, n_rep, axis=0)

    scale = 1.0 / np.sqrt(d)
    scores = np.matmul(q, k_rep.transpose(0, 2, 1)) * scale  # (H_q, n, n)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=-1, keepdims=True)
    return np.matmul(w, v_rep)


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []
    for H_q, H_kv, n, d in [
        (4, 4, 6, 8),   # MHA
        (4, 2, 6, 8),   # GQA, n_rep=2
        (8, 2, 5, 4),   # GQA, n_rep=4
        (6, 1, 7, 4),   # MQA
        (1, 1, 3, 4),   # degenerate single head
        (12, 3, 4, 6),  # GQA, n_rep=4
    ]:
        q = rng.normal(size=(H_q, n, d))
        k = rng.normal(size=(H_kv, n, d))
        v = rng.normal(size=(H_kv, n, d))
        scenarios.append((q, k, v))
    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for q, k, v in _scenarios():
        ref = _ref_gqa_attention(q, k, v)
        try:
            got = sol.gqa_broadcast_attention(q.copy(), k.copy(), v.copy())
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
