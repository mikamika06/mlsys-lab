import numpy as np


def _quantize_col(col, bits):
    qmax = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(col)) / qmax
    if scale == 0:
        return np.zeros_like(col)
    return np.clip(np.round(col / scale), -qmax, qmax) * scale


def gptq_quantize(W: np.ndarray, H_inv: np.ndarray, bits: int = 4) -> np.ndarray:
    # TODO: this is RTN because it skips the GPTQ remaining-column update.
    out = np.empty_like(W, dtype=np.float64)

    for j in range(W.shape[1]):
        out[:, j] = _quantize_col(W[:, j], bits)

    return out
