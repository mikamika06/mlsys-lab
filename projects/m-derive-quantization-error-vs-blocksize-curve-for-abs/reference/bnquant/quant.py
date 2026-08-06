import math
import numpy as np

def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def error_vs_blocksize(x, block_sizes):
    x_flat = x.flatten()
    n = len(x_flat)
    errors = []
    for bs in block_sizes:
        padded_len = ((n + bs - 1) // bs) * bs
        padded = np.zeros(padded_len, dtype=x.dtype)
        padded[:n] = x_flat
        reshaped = padded.reshape(-1, bs)
        maxs = np.max(np.abs(reshaped), axis=1, keepdims=True)
        maxs = np.where(maxs == 0, 1.0, maxs)
        scaled = np.clip(np.round(reshaped / maxs * 127.0), -127, 127)
        dequant = scaled * maxs / 127.0
        dequant_flat = dequant.flatten()[:n]
        mse = np.mean((x_flat - dequant_flat) ** 2)
        errors.append(float(mse))
    return errors

def blockwise_quantize_dequantize(x, block_size):
    shape = x.shape
    x_flat = x.flatten()
    n = len(x_flat)
    padded_len = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_len, dtype=x.dtype)
    padded[:n] = x_flat
    reshaped = padded.reshape(-1, block_size)
    maxs = np.max(np.abs(reshaped), axis=1, keepdims=True)
    maxs = np.where(maxs == 0, 1.0, maxs)
    scaled = np.clip(np.round(reshaped / maxs * 127.0), -127, 127).astype(np.int8)
    dequant = (scaled.astype(np.float32) * maxs / 127.0).reshape(-1)
    dequant = dequant[:n].reshape(shape)
    return scaled, dequant, maxs.flatten()

def expected_outliers(num_cols, num_rows, threshold=6.0):
    prob_exceed_scalar = 2.0 * (1.0 - _normal_cdf(threshold))
    prob_col_exceed = 1.0 - (1.0 - prob_exceed_scalar) ** num_rows
    expected_count = num_cols * prob_col_exceed
    return float(expected_count)
