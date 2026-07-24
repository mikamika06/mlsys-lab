import numpy as np


def _quantize_symmetric(K: np.ndarray, bits: int, group_axis: int) -> np.ndarray:
    """Symmetric per-group uniform quantizer, one scale per index along the
    axis NOT reduced (group_axis is the axis reduced over to find the group's
    amax, i.e. group_axis=0 -> one scale per column / per-channel;
    group_axis=1 -> one scale per row / per-token)."""
    qmax = 2 ** (bits - 1) - 1
    amax = np.max(np.abs(K), axis=group_axis, keepdims=True)
    scale = np.where(amax > 1e-12, amax / qmax, 1.0)
    q = np.clip(np.round(K / scale), -qmax, qmax)
    return q * scale


def compare_k_quant_granularity(K: np.ndarray, bits: int):
    """
    Compare per-channel vs per-token symmetric quantization of a key cache
    K (n_tokens, d_channels).

    Returns (mse_per_channel, mse_per_token):
      mse_per_channel: MSE when each COLUMN (channel) gets its own scale,
                        calibrated over all tokens (amax over axis=0).
      mse_per_token:   MSE when each ROW (token) gets its own scale,
                        calibrated over all channels (amax over axis=1).
    """
    K = np.asarray(K, dtype=np.float64)

    deq_channel = _quantize_symmetric(K, bits, group_axis=0)
    deq_token = _quantize_symmetric(K, bits, group_axis=1)

    mse_channel = float(np.mean((K - deq_channel) ** 2))
    mse_token = float(np.mean((K - deq_token) ** 2))
    return mse_channel, mse_token
