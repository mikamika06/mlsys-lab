import numpy as np


def _e4m3_roundtrip(x, scale):
    y = np.asarray(x, dtype=np.float64) / scale
    y = np.clip(y, -448.0, 448.0)
    sign = np.sign(y)
    ay = np.abs(y)
    if np.any(ay):
        exp = np.floor(np.log2(np.maximum(ay, 2 ** -9)))
        exp = np.clip(exp, -6, 7)
        frac = ay / (2.0 ** exp) - 1.0
        mant = np.round(frac * 8.0) / 8.0
        val = (1.0 + mant) * (2.0 ** exp)
        val = np.where(ay < 2 ** -6, np.round(ay / (2 ** -9)) * (2 ** -9), val)
    else:
        val = ay
    return sign * val * scale


def _oracle(Q, K, V):
    sk = max(np.max(np.abs(K)) / 448.0, 1e-12)
    sv = max(np.max(np.abs(V)) / 448.0, 1e-12)
    K2 = _e4m3_roundtrip(K, sk)
    V2 = _e4m3_roundtrip(V, sv)
    logits = Q @ K2.T / np.sqrt(Q.shape[1])
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs @ V2


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -0.5, 0.25]], dtype=np.float64),
            np.array([[500.0, -300.0, 2.0], [-450.0, 200.0, -1.0]], dtype=np.float64),
            np.array([[10.0, -3.0, 7.0], [4.0, 8.0, -6.0]], dtype=np.float64),
        ),
        (
            np.array([[2.0, 1.0], [-1.0, 0.5]], dtype=np.float64),
            np.array([[700.0, 50.0], [-600.0, -20.0], [100.0, 300.0]], dtype=np.float64),
            np.array([[20.0, 1.0], [5.0, -8.0], [-4.0, 9.0]], dtype=np.float64),
        ),
    ]
    worst = 0.0
    for Q, K, V in cases:
        try:
            got = np.asarray(sol.fp8_attention_output(Q, K, V), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(Q, K, V)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
