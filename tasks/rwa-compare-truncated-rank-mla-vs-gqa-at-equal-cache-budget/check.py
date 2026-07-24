import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _attention(Q, K, V):
    d = Q.shape[-1]
    Qh = Q.transpose(0, 2, 1, 3)
    Kh = K.transpose(0, 2, 1, 3)
    Vh = V.transpose(0, 2, 1, 3)
    scores = (Qh @ Kh.swapaxes(-2, -1)) / np.sqrt(d)
    weights = _softmax(scores, axis=-1)
    out = weights @ Vh
    return out.transpose(0, 2, 1, 3)


def _gqa_output(Q, K, V, g):
    batch, seq_k, n_heads, d = K.shape
    n_kv = n_heads // g
    Kg = K.reshape(batch, seq_k, n_kv, g, d).mean(axis=3)
    Vg = V.reshape(batch, seq_k, n_kv, g, d).mean(axis=3)
    K_bc = np.repeat(Kg, g, axis=2)
    V_bc = np.repeat(Vg, g, axis=2)
    return _attention(Q, K_bc, V_bc)


def _mla_output(Q, K, V, rank):
    batch, seq_k, n_heads, d = K.shape
    Kf = K.reshape(batch, seq_k, n_heads * d)
    Vf = V.reshape(batch, seq_k, n_heads * d)
    M = np.concatenate([Kf, Vf], axis=-1)  # (batch, seq_k, 2*n_heads*d)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, S.shape[-1])
    Ur = U[..., :, :r]
    Sr = S[..., :r]
    Vtr = Vt[..., :r, :]
    M_rec = (Ur * Sr[..., None, :]) @ Vtr

    Kr = M_rec[..., : n_heads * d].reshape(batch, seq_k, n_heads, d)
    Vr = M_rec[..., n_heads * d :].reshape(batch, seq_k, n_heads, d)
    return _attention(Q, Kr, Vr)


def _oracle(Q, K, V, group_size):
    n_heads = Q.shape[2]
    d = Q.shape[3]
    n_kv = n_heads // group_size
    rank = 2 * n_kv * d  # equal per-token cache budget vs GQA(group_size)

    mha_out = _attention(Q, K, V)
    gqa_out = _gqa_output(Q, K, V, group_size)
    mla_out = _mla_output(Q, K, V, rank)

    gqa_err = float(np.max(np.abs(gqa_out - mha_out)))
    mla_err = float(np.max(np.abs(mla_out - mha_out)))
    winner = "mla" if mla_err < gqa_err else "gqa"
    return gqa_err, mla_err, winner


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    # (batch, seq_k, n_heads, head_dim, group_size)
    cases = [
        (2, 48, 8, 4, 2),
        (2, 48, 8, 4, 4),
        (2, 48, 8, 4, 8),
        (1, 40, 6, 4, 2),
        (1, 40, 6, 4, 3),
        (1, 40, 6, 4, 6),
        (3, 50, 4, 8, 2),
        (3, 50, 4, 8, 4),
    ]

    max_err = 0.0
    winner_ok = 1.0

    for batch, seq_k, n_heads, d, group_size in cases:
        Q = rng.standard_normal((batch, seq_k, n_heads, d))
        K = rng.standard_normal((batch, seq_k, n_heads, d))
        V = rng.standard_normal((batch, seq_k, n_heads, d))

        ref_gqa_err, ref_mla_err, ref_winner = _oracle(Q, K, V, group_size)

        try:
            got = sol.mla_gqa_equal_budget_compare(Q.copy(), K.copy(), V.copy(), group_size)
        except Exception:
            return {"max_abs_err": float("inf"), "exact_match": 0.0}

        try:
            got_gqa_err, got_mla_err, got_winner = got
            got_gqa_err = float(got_gqa_err)
            got_mla_err = float(got_mla_err)
        except Exception:
            return {"max_abs_err": float("inf"), "exact_match": 0.0}

        max_err = max(max_err, abs(got_gqa_err - ref_gqa_err), abs(got_mla_err - ref_mla_err))
        if got_winner != ref_winner:
            winner_ok = 0.0

    return {"max_abs_err": max_err, "exact_match": winner_ok}
