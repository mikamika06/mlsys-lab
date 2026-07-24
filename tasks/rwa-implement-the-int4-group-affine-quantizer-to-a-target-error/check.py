import numpy as np

from mlsys import scorers


def _qd_1d(x: np.ndarray, bits: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return x.copy()
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    zero = min(max(zero, 0), qmax)
    codes = np.clip(np.round(x / scale + zero), 0, qmax)
    return (codes - zero) * scale


def _grouped_dequant(x: np.ndarray, group_size: int, bits: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    rows, cols = x.shape
    out = np.empty_like(x)
    for r in range(rows):
        row = x[r]
        for s in range(0, cols, group_size):
            seg = row[s:s + group_size]
            out[r, s:s + group_size] = _qd_1d(seg, bits)
    return out


def _attn(Q, K, V):
    d = Q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    scores = (Q @ K.T) * scale
    scores = scores - np.max(scores, axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=1, keepdims=True)
    return w @ V


def _build_mse_tensor() -> np.ndarray:
    rng = np.random.default_rng(0)
    rows, cols = 16, 256
    W = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    flat = W.reshape(-1)
    idx = rng.choice(flat.size, size=10, replace=False)
    flat[idx] += rng.normal(loc=0.0, scale=10.0, size=10)
    return flat.reshape(rows, cols)


def _build_attn_tensors():
    rng = np.random.default_rng(1)
    n, d = 20, 64
    Q = rng.normal(size=(n, d))
    K = rng.normal(size=(n, d))
    V = rng.normal(size=(n, d))
    return Q, K, V


def grade(sol, fx) -> dict:
    W = _build_mse_tensor()
    worst_mse = 0.0
    for group_size in (32, 64, 128):
        ref = _grouped_dequant(W, group_size)
        try:
            got = sol.quantize_dequantize_int4_grouped(W.copy(), group_size)
        except Exception:
            return {"mse": float("inf"), "attn_max_abs_err": float("inf")}

        try:
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"mse": float("inf"), "attn_max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"mse": float("inf"), "attn_max_abs_err": float("inf")}

        m = scorers.mse(ref, got)
        if not np.isfinite(m):
            return {"mse": float("inf"), "attn_max_abs_err": float("inf")}
        worst_mse = max(worst_mse, m)

    Q, K, V = _build_attn_tensors()
    ref_attn = _attn(Q, K, V)
    try:
        K_hat = sol.quantize_dequantize_int4_grouped(K.copy(), 32)
        V_hat = sol.quantize_dequantize_int4_grouped(V.copy(), 32)
    except Exception:
        return {"mse": worst_mse, "attn_max_abs_err": float("inf")}

    try:
        K_hat = np.asarray(K_hat, dtype=np.float64)
        V_hat = np.asarray(V_hat, dtype=np.float64)
        got_attn = _attn(Q, K_hat, V_hat)
    except Exception:
        return {"mse": worst_mse, "attn_max_abs_err": float("inf")}

    if got_attn.shape != ref_attn.shape:
        return {"mse": worst_mse, "attn_max_abs_err": float("inf")}

    attn_err = scorers.max_abs_err(ref_attn, got_attn)
    if not np.isfinite(attn_err):
        attn_err = float("inf")

    return {"mse": worst_mse, "attn_max_abs_err": attn_err}
