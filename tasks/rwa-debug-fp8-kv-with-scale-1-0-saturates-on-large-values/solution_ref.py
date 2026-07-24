import numpy as np


def _e4m3_roundtrip(x, scale):
    y = np.asarray(x, dtype=np.float64) / scale
    y = np.clip(y, -448.0, 448.0)
    sign = np.sign(y)
    ay = np.abs(y)
    exp = np.floor(np.log2(np.maximum(ay, 2 ** -9)))
    exp = np.clip(exp, -6, 7)
    frac = ay / (2.0 ** exp) - 1.0
    mant = np.round(frac * 8.0) / 8.0
    val = (1.0 + mant) * (2.0 ** exp)
    val = np.where(ay < 2 ** -6, np.round(ay / (2 ** -9)) * (2 ** -9), val)
    val = np.where(ay == 0, 0.0, val)
    return sign * val * scale


def fp8_attention_output(Q, K, V):
    sk = max(float(np.max(np.abs(K))) / 448.0, 1e-12)
    sv = max(float(np.max(np.abs(V))) / 448.0, 1e-12)
    K_hat = _e4m3_roundtrip(K, sk)
    V_hat = _e4m3_roundtrip(V, sv)
    scores = Q @ K_hat.T / np.sqrt(Q.shape[1])
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return (probs @ V_hat).astype(np.float64)
