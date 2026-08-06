import numpy as np
from mlsys import scorers

def _reference_mha(X, Wq, Wk, Wv, Wo):
    batch, seq_len, d_model = X.shape
    H = 4
    head_dim = d_model // H
    assert d_model % H == 0
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    # reshape to heads: (batch, seq_len, H, head_dim)
    Qh = Q.reshape(batch, seq_len, H, head_dim).transpose(0, 2, 1, 3)
    Kh = K.reshape(batch, seq_len, H, head_dim).transpose(0, 2, 1, 3)
    Vh = V.reshape(batch, seq_len, H, head_dim).transpose(0, 2, 1, 3)
    # scaled dot‑product attention
    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn /= np.sum(attn, axis=-1, keepdims=True)
    out_h = attn @ Vh
    # merge heads and project
    out = out_h.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return out @ Wo

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(seed=42)
    batch, seq_len, d_model = 2, 4, 8
    H = 4
    assert d_model % H == 0
    X  = rng.standard_normal((batch, seq_len, d_model))
    Wq = rng.standard_normal((d_model, d_model))
    Wk = rng.standard_normal((d_model, d_model))
    Wv = rng.standard_normal((d_model, d_model))
    Wo = rng.standard_normal((d_model, d_model))

    X_list = X.tolist()
    Wq_list = Wq.tolist()
    Wk_list = Wk.tolist()
    Wv_list = Wv.tolist()
    Wo_list = Wo.tolist()

    try:
        got_list = sol.mha_forward(X_list, Wq_list, Wk_list, Wv_list, Wo_list)
        got = np.array(got_list)
    except Exception:
        return {"max_abs_err": float("inf")}
    ref = _reference_mha(X, Wq, Wk, Wv, Wo)
    err = scorers.max_abs_err(ref, got)
    return {"max_abs_err": err}
