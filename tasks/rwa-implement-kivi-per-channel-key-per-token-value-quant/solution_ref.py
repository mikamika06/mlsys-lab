import numpy as np


def _affine_quant_dequant(x: np.ndarray, bits: int, axis) -> np.ndarray:
    """Uniform affine (asymmetric) min-max quantizer/dequantizer.

    `axis` selects the grouping: the min/max (and therefore the
    scale/zero-point) are computed per-group along that axis, and every
    element in the group shares that one scale/zero-point. `axis=None`
    means a single group for the whole tensor (per-tensor quant).
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    scale = (xmax - xmin) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    zero_point = np.round(-xmin / scale)
    q = np.clip(np.round(x / scale + zero_point), 0, qmax)
    return (q - zero_point) * scale


def _attention(K: np.ndarray, V: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = K.shape[1]
    logits = (K @ q) / np.sqrt(d)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / np.sum(w)
    return w @ V


def kivi_quant_errors(K: np.ndarray, V: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    """
    K, V: (n_tokens, d) fp64 key/value cache. q: (d,) fp64 query.
    bits: quantizer bit-width.

    KIVI quantizes keys PER-CHANNEL (one scale/zero-point per column,
    computed across all tokens) and values PER-TOKEN (one scale/zero-point
    per row, computed across all channels) -- the asymmetric axis choice
    that makes low-bit KV cache quantization viable, because RoPE-rotated
    key channels have consistent per-channel outlier structure while value
    outliers are token-specific.

    Returns np.array([
        k_mse_per_channel,   # MSE of per-channel-quantized K vs true K
        k_mse_per_tensor,    # MSE of a per-tensor-quantized K baseline vs true K
        attn_max_abs_err,    # max abs error of attention(K,V,q) using the
                              # (per-channel K, per-token V) KIVI-quantized
                              # cache, vs the exact fp64 attention output
    ])
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    K_per_channel = _affine_quant_dequant(K, bits, axis=0)
    V_per_token = _affine_quant_dequant(V, bits, axis=1)
    K_per_tensor = _affine_quant_dequant(K, bits, axis=None)

    k_mse_per_channel = float(np.mean((K_per_channel - K) ** 2))
    k_mse_per_tensor = float(np.mean((K_per_tensor - K) ** 2))

    base = _attention(K, V, q)
    kivi_out = _attention(K_per_channel, V_per_token, q)
    attn_max_abs_err = float(np.max(np.abs(kivi_out - base)))

    return np.array([k_mse_per_channel, k_mse_per_tensor, attn_max_abs_err])
