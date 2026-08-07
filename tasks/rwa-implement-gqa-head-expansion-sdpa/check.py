import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _repeat_kv(x, n_rep):
    # (batch, seq, n_kv, d) -> (batch, seq, n_kv * n_rep, d), repeat_interleave
    return np.repeat(x, n_rep, axis=2)


def _oracle(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n_q = Q.shape[2]
    n_kv = K.shape[2]
    n_rep = n_q // n_kv

    K_exp = _repeat_kv(K, n_rep)
    V_exp = _repeat_kv(V, n_rep)

    d = Q.shape[-1]
    Qh = Q.transpose(0, 2, 1, 3)
    Kh = K_exp.transpose(0, 2, 1, 3)
    Vh = V_exp.transpose(0, 2, 1, 3)

    scores = (Qh @ Kh.swapaxes(-2, -1)) / np.sqrt(d)
    weights = _softmax(scores, axis=-1)
    out = (weights @ Vh).transpose(0, 2, 1, 3)

    memory_ratio = n_kv / n_q
    return out, memory_ratio


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    # (batch, seq_q, seq_k, n_q, n_kv, head_dim)
    cases = [
        (2, 5, 5, 4, 4, 8),    # degenerate: n_kv == n_q -> plain MHA
        (1, 4, 4, 8, 2, 4),    # r = 4
        (3, 3, 6, 6, 3, 16),   # r = 2
        (2, 2, 2, 12, 1, 4),   # MQA extreme: r = n_q
        (1, 5, 3, 6, 2, 8),    # asymmetric seq lens, r = 3
    ]

    max_err = 0.0
    ratio_ok = 1.0

    for batch, seq_q, seq_k, n_q, n_kv, d in cases:
        Q = rng.standard_normal((batch, seq_q, n_q, d))
        K = rng.standard_normal((batch, seq_k, n_kv, d))
        V = rng.standard_normal((batch, seq_k, n_kv, d))

        ref_out, ref_ratio = _oracle(Q, K, V)

        try:
            got = sol.gqa_head_expansion_attention(Q.tolist(), K.tolist(), V.tolist())
            got_out, got_ratio = got
            got_out = np.asarray(got_out, dtype=np.float64)
            got_ratio = float(got_ratio)
        except Exception:
            return {"max_abs_err": float("inf"), "size_ratio": 0.0}

        if got_out.shape != ref_out.shape:
            return {"max_abs_err": float("inf"), "size_ratio": 0.0}

        max_err = max(max_err, float(np.max(np.abs(got_out - ref_out))))
        if got_ratio != ref_ratio:
            ratio_ok = 0.0

    return {"max_abs_err": max_err, "size_ratio": ratio_ok}
