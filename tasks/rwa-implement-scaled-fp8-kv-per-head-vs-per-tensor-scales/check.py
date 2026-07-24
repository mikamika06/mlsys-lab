import numpy as np


def _e4m3(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    ax = np.abs(x)
    nz = ax > 0
    if np.any(nz):
        vals = ax[nz]
        vals = np.minimum(vals, 448.0)
        exp = np.floor(np.log2(vals))
        exp = np.maximum(exp, -6)
        base = np.power(2.0, exp)
        mant = vals / base - 1.0
        mant_q = np.round(mant * 8.0) / 8.0
        vals_q = base * (1.0 + mant_q)
        carry = vals_q > 448.0
        vals_q = np.minimum(vals_q, 448.0)
        out[nz] = np.sign(x[nz]) * vals_q
    return out


def _quant_dequant(x, per_head):
    x = np.asarray(x, dtype=np.float64)
    if per_head:
        scale = np.max(np.abs(x), axis=(1, 2), keepdims=True) / 448.0
    else:
        scale = np.max(np.abs(x)) / 448.0
    scale = np.maximum(scale, 1e-12)
    return _e4m3(x / scale) * scale


def _oracle(K, V, Q, per_head):
    Kd = _quant_dequant(K, per_head)
    Vd = _quant_dequant(V, per_head)
    scores = np.matmul(Q.astype(np.float64), np.swapaxes(Kd, 1, 2))
    scores = scores / np.sqrt(K.shape[-1])
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, Vd)


def _reconstruction_error(x, per_head):
    xd = _quant_dequant(x, per_head)
    return float(np.mean(np.abs(xd - x)))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    K = rng.normal(0, 1, size=(3, 9, 8))
    V = rng.normal(0, 1, size=(3, 9, 8))
    Q = rng.normal(0, 1, size=(3, 4, 8))

    try:
        got = sol.scaled_fp8_kv_attention(K, V, Q, True)
    except Exception:
        return {"max_abs_err": float("inf"), "per_head_advantage": 0.0}

    ref = _oracle(K, V, Q, True)
    err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))

    K2 = rng.normal(0, 1, size=(3, 16, 8))
    V2 = rng.normal(0, 1, size=(3, 16, 8))
    K2[2] *= 200.0
    V2[2] *= 200.0
    head_err = _reconstruction_error(V2, True)
    tensor_err = _reconstruction_error(V2, False)
    advantage = 1.0 if head_err < tensor_err else 0.0

    return {
        "max_abs_err": err,
        "per_head_advantage": advantage
    }
