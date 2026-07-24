import numpy as np


def _quantize_column(x, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(x)) / qmax
    if scale == 0:
        return np.zeros_like(x)
    codes = np.clip(np.round(x / scale), -qmax, qmax)
    return codes * scale


def gptq_quantize(W, H, bits=4):
    work = np.asarray(W, dtype=np.float64).copy()
    out = np.zeros_like(work)

    for j in range(work.shape[1]):
        current = work[:, j].copy()
        quantized = _quantize_column(current, bits)
        out[:, j] = quantized

        # TODO: propagate the quantization error into future columns.
        # This version leaves later columns unchanged, so it behaves like
        # independent column quantization instead of GPTQ error compensation.

    return out
