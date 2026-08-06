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


def mla_gqa_equal_budget_compare(Q: np.ndarray, K: np.ndarray, V: np.ndarray, group_size: int):
    """
    Compare a GQA(group_size) reconstruction and a rank-matched truncated
    MLA reconstruction against the true full-MHA output, at equal per-token
    cache budget.

    Returns (gqa_err, mla_err, winner) where winner is "mla" if
    mla_err < gqa_err else "gqa".
    """
    batch, seq_k, n_heads, d = K.shape
    n_kv = n_heads // group_size

    mha_out = _attention(Q, K, V)

    # --- GQA(group_size): mean-pool K/V within each group, broadcast back.
    Kg = K.reshape(batch, seq_k, n_kv, group_size, d).mean(axis=3)
    Vg = V.reshape(batch, seq_k, n_kv, group_size, d).mean(axis=3)
    K_bc = np.repeat(Kg, group_size, axis=2)
    V_bc = np.repeat(Vg, group_size, axis=2)
    gqa_out = _attention(Q, K_bc, V_bc)

    # --- MLA(rank): rank-matched to GQA's per-token cache budget:
    # GQA stores 2 * n_kv * d scalars per token (K and V per kv head);
    # MLA stores `rank` scalars per token (one shared low-rank latent).
    rank = 2 * n_kv * d

    Kf = K.reshape(batch, seq_k, n_heads * d)
    Vf = V.reshape(batch, seq_k, n_heads * d)
    M = np.concatenate([Kf, Vf], axis=-1)  # (batch, seq_k, 2*n_heads*d)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, S.shape[-1])
    Ur = U[..., :, :r]
    Sr = S[..., :r]
    Vtr = Vt[..., :r, :]
    M_rec = (Ur * Sr[..., None, :]) @ Vtr  # best rank-r approximation (Eckart-Young)

    Kr = M_rec[..., : n_heads * d].reshape(batch, seq_k, n_heads, d)
    Vr = M_rec[..., n_heads * d :].reshape(batch, seq_k, n_heads, d)
    mla_out = _attention(Q, Kr, Vr)

    gqa_err = float(np.max(np.abs(gqa_out - mha_out)))
    mla_err = float(np.max(np.abs(mla_out - mha_out)))
    winner = "mla" if mla_err < gqa_err else "gqa"
    return gqa_err, mla_err, winner
