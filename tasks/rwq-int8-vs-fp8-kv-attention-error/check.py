import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attention(Q, K, V):
    d = Q.shape[1]
    return _softmax(Q.astype(np.float64) @ K.astype(np.float64).T / np.sqrt(d)) @ V.astype(np.float64)


def _int8_quant(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        return np.zeros_like(x)
    return np.round(x / scale) * scale


def _fp8_e4m3_quant(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = x != 0
    ax = np.abs(x[nz])
    exp = np.floor(np.log2(ax))
    exp = np.clip(exp, -6, 7)
    scale = np.power(2.0, exp - 3)
    rounded = np.round(ax / scale) * scale
    rounded = np.minimum(rounded, 240.0)
    out[nz] = np.sign(x[nz]) * rounded
    return out


def _oracle(Q, K, V):
    ref = _attention(Q, K, V)
    k_i = _int8_quant(K)
    v_i = _int8_quant(V)
    k_f = _fp8_e4m3_quant(K)
    v_f = _fp8_e4m3_quant(V)
    int8_mse = float(np.mean((ref - _attention(Q, k_i, v_i)) ** 2))
    fp8_mse = float(np.mean((ref - _attention(Q, k_f, v_f)) ** 2))
    winner = "int8" if int8_mse <= fp8_mse else "fp8"
    return int8_mse, fp8_mse, winner


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.2, -0.1], [0.4, 0.3]], dtype=np.float32),
            np.array([[0.1, 0.5], [-0.2, 0.4]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
        (
            np.array([[1.1, -0.7, 0.2], [0.4, 0.9, -1.3], [0.0, 0.2, 0.5]], dtype=np.float32),
            np.array([[0.5, -0.2, 1.0], [-0.8, 0.3, 0.4], [0.6, 0.7, -0.1]], dtype=np.float32),
            np.array([[1.0, 2.0], [-1.0, 0.5], [0.2, -0.4]], dtype=np.float32),
        ),
        (
            np.arange(12, dtype=np.float32).reshape(3, 4) / 5.0,
            np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0 - 0.5,
            np.arange(6, dtype=np.float32).reshape(3, 2) / 3.0,
        ),
    ]

    mse_ok = 1.0
    winner_ok = 1.0
    for Q, K, V in cases:
        ref_i, ref_f, ref_w = _oracle(Q, K, V)
        try:
            got_i, got_f, got_w = sol.kv_attention_quant_error(Q, K, V)
        except Exception:
            return {"mse": 0.0, "winner_match": 0.0}
        if not (abs(float(got_i) - ref_i) <= 1e-6 and abs(float(got_f) - ref_f) <= 1e-6):
            mse_ok = 0.0
        if got_w != ref_w:
            winner_ok = 0.0
    return {"mse": mse_ok, "winner_match": winner_ok}
