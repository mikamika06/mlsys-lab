import numpy as np

def nf4_quantize_dequantize(x):
    quantiles = np.array([
        -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343,
        -0.09105004, 0.0, 0.079580, 0.160930, 0.246112, 0.337915,
        0.440709, 0.562617, 0.722956, 1.0
    ], dtype=np.float32)
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return x.copy()
    x_norm = x / max_val
    idx = np.abs(x_norm[..., np.newaxis] - quantiles).argmin(axis=-1)
    x_quant = quantiles[idx] * max_val
    return x_quant

def measure_nf4_error(x, cycles):
    curr = x.copy()
    errors = []
    for _ in range(cycles):
        nxt = nf4_quantize_dequantize(curr)
        err = np.mean(np.abs(curr - nxt))
        errors.append(err)
        curr = nxt
    return errors
