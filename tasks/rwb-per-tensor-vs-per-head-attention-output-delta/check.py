import numpy as np


def _e4m3(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    ax = np.abs(x)
    nz = ax > 0
    if np.any(nz):
        vals = np.minimum(ax[nz], 448.0)
        exp = np.maximum(np.floor(np.log2(vals)), -6)
        base = np.power(2.0, exp)
        mant = vals / base - 1.0
        mant_q = np.round(mant * 8.0) / 8.0
        vals_q = np.minimum(base * (1.0 + mant_q), 448.0)
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


def _attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, 1, 2)) / np.sqrt(d)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, V)


def _per_head_delta(O_ref, O_quant):
    num = np.linalg.norm((O_quant - O_ref).reshape(O_ref.shape[0], -1), axis=1)
    den = np.linalg.norm(O_ref.reshape(O_ref.shape[0], -1), axis=1) + 1e-12
    return float(np.mean(num / den))


def _oracle(Q, K, V):
    O_ref = _attention(Q, K, V)

    K_pt = _quant_dequant(K, per_head=False)
    V_pt = _quant_dequant(V, per_head=False)
    O_pt = _attention(Q, K_pt, V_pt)

    K_ph = _quant_dequant(K, per_head=True)
    V_ph = _quant_dequant(V, per_head=True)
    O_ph = _attention(Q, K_ph, V_ph)

    return _per_head_delta(O_ref, O_pt), _per_head_delta(O_ref, O_ph)


def grade(sol, fx) -> dict:
    Q, K, V = fx["q"], fx["k"], fx["v"]

    ref_pt, ref_ph = _oracle(Q, K, V)

    try:
        got = sol.kv_scale_granularity_delta(Q.copy(), K.copy(), V.copy())
        got_pt, got_ph = float(got[0]), float(got[1])
    except Exception:
        return {"rel_err": float("inf")}

    if not (got_ph < got_pt):
        return {"rel_err": float("inf")}

    discrepancy = max(abs(got_pt - ref_pt), abs(got_ph - ref_ph))
    return {"rel_err": discrepancy}
