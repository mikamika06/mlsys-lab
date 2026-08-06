import numpy as np


def _quantize_col(col, bits):
    qmax = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(col)) / qmax
    if scale == 0:
        return np.zeros_like(col)
    codes = np.clip(np.round(col / scale), -qmax, qmax)
    return codes * scale


def gptq_quantize(W: np.ndarray, H_inv: np.ndarray, bits: int = 4) -> np.ndarray:
    work = np.array(W, dtype=np.float64, copy=True)
    result = np.zeros_like(work)
    cols = work.shape[1]

    for j in range(cols):
        current = work[:, j].copy()
        quantized = _quantize_col(current, bits)
        result[:, j] = quantized
        error = quantized - current

        for k in range(j + 1, cols):
            work[:, k] += error * (H_inv[j, k] / H_inv[j, j])

    return result
