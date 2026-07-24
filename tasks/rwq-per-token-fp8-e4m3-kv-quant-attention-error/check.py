import numpy as np


def _fp8_e4m3_round(x):
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    out = np.zeros_like(ax)

    normal = ax >= 2 ** -6
    if np.any(normal):
        a = ax[normal]
        exp = np.floor(np.log2(a))
        exp = np.clip(exp, -6, 8)
        step = 2.0 ** (exp - 3)
        mant = np.round(a / step) * step
        mant = np.minimum(mant, 448.0)
        out[normal] = mant

    sub = ~normal
    if np.any(sub):
        out[sub] = np.round(ax[sub] / (2 ** -9)) * (2 ** -9)

    return sign * out


def _quant_rows(x):
    scales = np.max(np.abs(x), axis=1, keepdims=True) / 448.0
    scales = np.where(scales == 0, 1.0, scales)
    q = _fp8_e4m3_round(x / scales)
    return q * scales


def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _oracle(Q, K, V):
    Kq = _quant_rows(K)
    Vq = _quant_rows(V)
    full = _softmax(Q @ K.T / np.sqrt(Q.shape[1])) @ V
    fp8 = _softmax(Q @ Kq.T / np.sqrt(Q.shape[1])) @ Vq
    mse = float(np.mean((fp8 - full) ** 2))
    return fp8, mse


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1234)
    cases = [
        (
            rng.normal(size=(3, 4)),
            rng.normal(size=(5, 4)),
            rng.normal(size=(5, 2)),
        ),
        (
            rng.normal(size=(2, 8)),
            rng.normal(size=(7, 8)),
            rng.normal(size=(7, 3)),
        ),
        (
            np.zeros((2, 4)),
            rng.normal(size=(4, 4)),
            rng.normal(size=(4, 1)),
        ),
    ]

    max_err = 0.0
    max_mse = 0.0
    for Q, K, V in cases:
        ref_y, ref_mse = _oracle(Q, K, V)
        try:
            got_y, got_mse = sol.fp8_kv_attention(Q, K, V)
        except Exception:
            return {"rel_err": 1.0, "mse": 1.0}
        err = np.linalg.norm(np.asarray(got_y) - ref_y) / (
            np.linalg.norm(ref_y) + 1e-12
        )
        max_err = max(max_err, float(err))
        max_mse = max(max_mse, float(abs(float(got_mse) - ref_mse)))

    return {"rel_err": max_err, "mse": max_mse}
