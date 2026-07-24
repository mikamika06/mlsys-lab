import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attention(Q, K, V):
    return _softmax(Q.astype(np.float64) @ K.astype(np.float64).T / np.sqrt(Q.shape[1])) @ V.astype(np.float64)


def _int8_quant(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        return np.zeros_like(x)
    return np.round(x / scale) * scale


def _fp8_e4m3_quant(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask = x != 0
    ax = np.abs(x[mask])
    exp = np.clip(np.floor(np.log2(ax)), -6, 7)
    step = np.power(2.0, exp - 3)
    out[mask] = np.sign(x[mask]) * np.minimum(np.round(ax / step) * step, 240.0)
    return out


def kv_attention_quant_error(Q, K, V):
    ref = _attention(Q, K, V)
    int8_out = _attention(Q, _int8_quant(K), _int8_quant(V))
    fp8_out = _attention(Q, _fp8_e4m3_quant(K), _fp8_e4m3_quant(V))
    int8_mse = float(np.mean((ref - int8_out) ** 2))
    fp8_mse = float(np.mean((ref - fp8_out) ** 2))
    return int8_mse, fp8_mse, "int8" if int8_mse <= fp8_mse else "fp8"
