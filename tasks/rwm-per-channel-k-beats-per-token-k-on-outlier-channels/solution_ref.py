import numpy as np


def _quantize_symmetric(K: np.ndarray, bits: int, group_axis: int) -> np.ndarray:
    """Symmetric per-group uniform quantizer, one scale per index along the
    axis NOT reduced (group_axis is the axis reduced over to find the group's
    amax, i.e. group_axis=0 -> one scale per column / per-channel;
    group_axis=1 -> one scale per row / per-token)."""
    qmax = float(2 ** (bits - 1) - 1)
    rows, cols = K.shape
    out = np.zeros((rows, cols), dtype=K.dtype)

    if group_axis == 0:
        for j in range(cols):
            max_val = 0.0
            for i in range(rows):
                val = float(K[i, j])
                val_abs = -val if val < 0.0 else val
                if val_abs > max_val:
                    max_val = val_abs
            scale = max_val / qmax if max_val > 1e-12 else 1.0
            for i in range(rows):
                val = float(K[i, j])
                scaled = val / scale
                rnd = round(scaled)
                if rnd < -qmax:
                    clipped = -qmax
                elif rnd > qmax:
                    clipped = qmax
                else:
                    clipped = float(rnd)
                out[i, j] = clipped * scale
    else:
        for i in range(rows):
            max_val = 0.0
            for j in range(cols):
                val = float(K[i, j])
                val_abs = -val if val < 0.0 else val
                if val_abs > max_val:
                    max_val = val_abs
            scale = max_val / qmax if max_val > 1e-12 else 1.0
            for j in range(cols):
                val = float(K[i, j])
                scaled = val / scale
                rnd = round(scaled)
                if rnd < -qmax:
                    clipped = -qmax
                elif rnd > qmax:
                    clipped = qmax
                else:
                    clipped = float(rnd)
                out[i, j] = clipped * scale

    return out


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
    rows, cols = K.shape
    total_elem = float(rows * cols)

    deq_channel = _quantize_symmetric(K, bits, group_axis=0)
    deq_token = _quantize_symmetric(K, bits, group_axis=1)

    sum_sq_channel = 0.0
    sum_sq_token = 0.0

    for i in range(rows):
        for j in range(cols):
            diff_c = float(K[i, j]) - float(deq_channel[i, j])
            sum_sq_channel += diff_c * diff_c

            diff_t = float(K[i, j]) - float(deq_token[i, j])
            sum_sq_token += diff_t * diff_t

    mse_channel = float(sum_sq_channel / total_elem)
    mse_token = float(sum_sq_token / total_elem)

    return mse_channel, mse_token
