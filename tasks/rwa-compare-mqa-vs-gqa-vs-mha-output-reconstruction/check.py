import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _attention(Q, K, V):
    """Standard scaled dot-product attention, already-broadcast K/V.

    Q, K, V: (batch, seq, n_heads, head_dim); K/V share n_heads with Q.
    Returns (batch, seq, n_heads, head_dim).
    """
    d = Q.shape[-1]
    Qh = Q.transpose(0, 2, 1, 3)
    Kh = K.transpose(0, 2, 1, 3)
    Vh = V.transpose(0, 2, 1, 3)
    scores = (Qh @ Kh.swapaxes(-2, -1)) / np.sqrt(d)
    weights = _softmax(scores, axis=-1)
    out = weights @ Vh
    return out.transpose(0, 2, 1, 3)


def _grouped_kv(K, V, group_size):
    batch, seq_k, n_heads, d = K.shape
    n_kv = n_heads // group_size
    Kg = K.reshape(batch, seq_k, n_kv, group_size, d).mean(axis=3)
    Vg = V.reshape(batch, seq_k, n_kv, group_size, d).mean(axis=3)
    K_bc = np.repeat(Kg, group_size, axis=2)
    V_bc = np.repeat(Vg, group_size, axis=2)
    return K_bc, V_bc, n_kv


def _oracle(Q, K, V, group_sizes):
    n_heads = Q.shape[2]
    out = []
    for g in group_sizes:
        K_bc, V_bc, n_kv = _grouped_kv(K, V, g)
        out.append((_attention(Q, K_bc, V_bc), n_kv / n_heads))
    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    # (batch, seq, n_heads, head_dim, group_sizes)
    cases = [
        (2, 5, 8, 4, [1, 2, 4, 8]),      # MHA, two GQA arities, MQA
        (1, 4, 6, 8, [1, 3, 6]),
        (3, 3, 4, 16, [1, 2, 4]),
        (2, 2, 12, 4, [1, 4, 12]),
    ]

    max_err = 0.0
    ratio_ok = 1.0

    for batch, seq, n_heads, d, group_sizes in cases:
        Q = rng.standard_normal((batch, seq, n_heads, d))
        K = rng.standard_normal((batch, seq, n_heads, d))
        V = rng.standard_normal((batch, seq, n_heads, d))

        ref = _oracle(Q, K, V, group_sizes)

        try:
            got = sol.mha_gqa_mqa_reconstruct(Q.tolist(), K.tolist(), V.tolist(), list(group_sizes))
        except Exception:
            return {"max_abs_err": float("inf"), "size_ratio": 0.0}

        try:
            got = list(got)
        except Exception:
            return {"max_abs_err": float("inf"), "size_ratio": 0.0}

        if len(got) != len(ref):
            return {"max_abs_err": float("inf"), "size_ratio": 0.0}

        for (got_out, got_ratio), (ref_out, ref_ratio) in zip(got, ref):
            try:
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
