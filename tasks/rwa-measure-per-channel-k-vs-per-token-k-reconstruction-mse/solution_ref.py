import numpy as np


def _affine_quant_dequant(x: np.ndarray, bits: int, axis) -> np.ndarray:
    """Uniform affine (asymmetric) min-max quantizer/dequantizer.

    The min/max (and therefore the scale/zero-point) are computed
    per-group along `axis`; every element in a group shares that one
    scale/zero-point.
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


def per_channel_vs_per_token_k_mse(K: np.ndarray, bits: int) -> np.ndarray:
    """
    K: (n_tokens, d_channels) fp64 key cache. bits: quantizer bit-width.

    Quantize K two ways with a uniform affine min-max quantizer:
      - per-channel: one scale/zero-point per COLUMN, min/max taken across
        all tokens (axis=0).
      - per-token: one scale/zero-point per ROW, min/max taken across all
        channels (axis=1).

    Returns np.array([mse_per_channel, mse_per_token]), the reconstruction
    MSE (mean squared error of dequant(quant(K)) vs K) of each scheme.
    """
    K = np.asarray(K, dtype=np.float64)

    K_per_channel = _affine_quant_dequant(K, bits, axis=0)
    K_per_token = _affine_quant_dequant(K, bits, axis=1)

    mse_per_channel = float(np.mean((K_per_channel - K) ** 2))
    mse_per_token = float(np.mean((K_per_token - K) ** 2))

    return np.array([mse_per_channel, mse_per_token])
