import numpy as np


def _quantize_per_token_int8(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x), axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    codes = np.clip(np.round(x / scale), -127, 127)
    return codes * scale


def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _oracle(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    K_hat = _quantize_per_token_int8(K)
    V_hat = _quantize_per_token_int8(V)

    scale = np.sqrt(Q.shape[1])
    full = _softmax(Q @ K.T / scale) @ V
    out = _softmax(Q @ K_hat.T / scale) @ V_hat

    mse = float(np.mean((out - full) ** 2))
    return out, mse


def _cases():
    rng = np.random.default_rng(77)
    return [
        (rng.normal(size=(3, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 2))),
        (rng.normal(size=(6, 16)), rng.normal(size=(9, 16)), rng.normal(size=(9, 5))),
        (np.zeros((2, 4)), rng.normal(size=(4, 4)), rng.normal(size=(4, 1))),
        (rng.normal(size=(4, 8)) * 3.0, rng.normal(size=(4, 8)), np.zeros((4, 3))),
    ]


def grade(sol, fx) -> dict:
    max_rel = 0.0
    max_mse_err = 0.0

    for Q, K, V in _cases():
        ref_out, ref_mse = _oracle(Q, K, V)
        try:
            got_out, got_mse = sol.int8_kv_attention(
                np.array(Q, copy=True), np.array(K, copy=True), np.array(V, copy=True)
            )
            got_out = np.asarray(got_out, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf"), "mse": float("inf")}

        if got_out.shape != ref_out.shape or not np.all(np.isfinite(got_out)):
            return {"rel_err": float("inf"), "mse": float("inf")}

        rel = np.linalg.norm(got_out - ref_out) / (np.linalg.norm(ref_out) + 1e-12)
        max_rel = max(max_rel, float(rel))
        max_mse_err = max(max_mse_err, abs(float(got_mse) - ref_mse))

    return {"rel_err": max_rel, "mse": max_mse_err}
