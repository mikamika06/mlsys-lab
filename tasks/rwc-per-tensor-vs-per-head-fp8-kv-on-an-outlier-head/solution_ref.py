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
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, V)


def choose_kv_fp8_scheme(Q, K, V):
    ref = _attention(Q, K, V)

    tensor_out = _attention(Q, _quant_dequant(K), _quant_dequant(V))
    tensor_error = np.linalg.norm(tensor_out - ref) / (np.linalg.norm(ref) + 1e-12)

    head_out = _attention(
        Q,
        _quant_dequant(K, axis=(1, 2)),
        _quant_dequant(V, axis=(1, 2)),
    )
    head_error = np.linalg.norm(head_out - ref) / (np.linalg.norm(ref) + 1e-12)

    scheme = "per_head" if head_error < tensor_error else "per_tensor"
    return float(tensor_error), float(head_error), scheme
