import numpy as np

from mlsys import scorers


def _split_heads(A: np.ndarray, num_heads: int) -> np.ndarray:
    n, total = A.shape
    d_head = total // num_heads
    return A.reshape(n, num_heads, d_head).transpose(1, 0, 2)  # (H, n, d_head)


def _mha(Q: np.ndarray, K: np.ndarray, V: np.ndarray, num_heads: int) -> np.ndarray:
    n = Q.shape[0]
    Qh = _split_heads(Q, num_heads)
    Kh = _split_heads(K, num_heads)
    Vh = _split_heads(V, num_heads)
    d_head = Qh.shape[-1]
    scale = 1.0 / np.sqrt(d_head)
    scores = np.matmul(Qh, Kh.transpose(0, 2, 1)) * scale  # (H, n, n)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=-1, keepdims=True)
    out_h = np.matmul(w, Vh)  # (H, n, d_head)
    return out_h.transpose(1, 0, 2).reshape(n, num_heads * d_head)


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []
    for n, d_model, num_heads, d_head, r in [
        (5, 8, 2, 4, 6),
        (9, 16, 4, 8, 32),
        (3, 6, 1, 5, 3),
        (12, 20, 5, 4, 10),
        (7, 12, 3, 6, 18),
    ]:
        x = rng.normal(size=(n, d_model))
        W_Q = rng.normal(size=(d_model, num_heads * d_head))
        W_down_kv = rng.normal(size=(d_model, r))
        W_up_K = rng.normal(size=(r, num_heads * d_head))
        W_up_V = rng.normal(size=(r, num_heads * d_head))
        scenarios.append((x, W_Q, W_down_kv, W_up_K, W_up_V, num_heads, r))
    return scenarios


def grade(sol, fx) -> dict:
    worst_err = 0.0
    latent_ok = 1.0

    for x, W_Q, W_down_kv, W_up_K, W_up_V, num_heads, r in _scenarios():
        n = x.shape[0]
        c_kv_ref = x @ W_down_kv
        Q_ref = x @ W_Q
        K_ref = c_kv_ref @ W_up_K
        V_ref = c_kv_ref @ W_up_V
        out_ref = _mha(Q_ref, K_ref, V_ref, num_heads)

        try:
            out_got, c_kv_got = sol.mla_forward(
                x.copy(), W_Q.copy(), W_down_kv.copy(), W_up_K.copy(), W_up_V.copy(), num_heads,
            )
        except Exception:
            return {"max_abs_err": float("inf"), "latent_ok": 0.0}

        try:
            out_got = np.asarray(out_got, dtype=np.float64)
            c_kv_got = np.asarray(c_kv_got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "latent_ok": 0.0}

        if out_got.shape != out_ref.shape:
            return {"max_abs_err": float("inf"), "latent_ok": 0.0}

        err = scorers.max_abs_err(out_ref, out_got)
        if not np.isfinite(err):
            return {"max_abs_err": float("inf"), "latent_ok": 0.0}
        worst_err = max(worst_err, err)

        if c_kv_got.shape != (n, r):
            latent_ok = 0.0
        else:
            latent_err = scorers.max_abs_err(c_kv_ref, c_kv_got)
            if not np.isfinite(latent_err) or latent_err > 1e-8:
                latent_ok = 0.0

    return {"max_abs_err": worst_err, "latent_ok": latent_ok}
