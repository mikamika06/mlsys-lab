import numpy as np


def _quantize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    scales = np.max(np.abs(x), axis=1, keepdims=True) / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.round(x / scales).clip(-127, 127).astype(np.int8)
    return q.astype(np.float64) * scales


def _quantize_tensor(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        scale = 1.0
    q = np.round(x / scale).clip(-127, 127).astype(np.int8)
    return q.astype(np.float64) * scale


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _oracle(Q, K, V):
    d = K.shape[1]
    Kq = _quantize_rows(K)
    Vq = _quantize_rows(V)
    scores = Q @ Kq.T / np.sqrt(d)
    return _softmax(scores) @ Vq


def _buggy(Q, K, V):
    d = K.shape[1]
    Kq = _quantize_tensor(K)
    Vq = _quantize_tensor(V)
    scores = Q @ Kq.T / np.sqrt(d)
    return _softmax(scores) @ Vq


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    Q = rng.normal(size=(4, 8))
    K = rng.normal(size=(16, 8))
    V = rng.normal(size=(16, 8))

    try:
        got = np.asarray(sol.quantized_kv_attention(Q, K, V), dtype=np.float64)
    except Exception:
        return {"rel_err": 1.0, "mse_ratio_vs_buggy": 1e9}

    ref = _oracle(Q, K, V)
    bad = _buggy(Q, K, V)

    rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
    mse_good = float(np.mean((got - ref) ** 2))
    mse_bad = float(np.mean((bad - ref) ** 2))

    return {
        "rel_err": rel,
        "mse_ratio_vs_buggy": mse_good / (mse_bad + 1e-12),
    }
