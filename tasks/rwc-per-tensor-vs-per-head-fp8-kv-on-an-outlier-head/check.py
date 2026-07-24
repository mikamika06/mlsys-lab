import numpy as np


def _quant_dequant(x, axis=None):
    x = np.asarray(x, dtype=np.float64)
    if axis is None:
        scale = np.max(np.abs(x)) / 127.0
        if scale == 0:
            return np.zeros_like(x)
        return np.clip(np.round(x / scale), -127, 127) * scale
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.round(x / scale), -127, 127) * scale


def _attention(Q, K, V):
    logits = np.matmul(Q, np.swapaxes(K, 1, 2)) / np.sqrt(Q.shape[-1])
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, V)


def _oracle(Q, K, V):
    ref = _attention(Q, K, V)

    kt = _quant_dequant(K)
    vt = _quant_dequant(V)
    tensor_error = np.linalg.norm(_attention(Q, kt, vt) - ref) / (np.linalg.norm(ref) + 1e-12)

    kh = _quant_dequant(K, axis=(1, 2))
    vh = _quant_dequant(V, axis=(1, 2))
    head_error = np.linalg.norm(_attention(Q, kh, vh) - ref) / (np.linalg.norm(ref) + 1e-12)

    scheme = "per_head" if head_error < tensor_error else "per_tensor"
    return float(tensor_error), float(head_error), scheme


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    Q = rng.normal(size=(4, 8, 16))
    K = rng.normal(size=(4, 8, 16))
    V = rng.normal(size=(4, 8, 16))
    K[0] *= 40.0
    V[0] *= 40.0

    ref = _oracle(Q, K, V)
    try:
        got = sol.choose_kv_fp8_scheme(Q, K, V)
        tensor_error, head_error, scheme = got
        rel_err = float(
            max(
                abs(float(tensor_error) - ref[0]),
                abs(float(head_error) - ref[1]),
            )
        )
        scheme_match = 1.0 if scheme == ref[2] else 0.0
    except Exception:
        rel_err = 1.0
        scheme_match = 0.0

    return {
        "rel_err": rel_err,
        "scheme_match": scheme_match,
    }
