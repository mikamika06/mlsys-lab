import numpy as np


def quantize_per_channel(W: np.ndarray, n_bits: int = 8):
    """Symmetric per-output-channel (row-wise) integer quantization of ``W``.

    Returns ``(q, scale)`` with int8 codes and a ``(C_out, 1)`` scale.
    """
    qmax = 2 ** (n_bits - 1) - 1
    Wf = np.asarray(W, dtype=np.float64)
    amax = np.max(np.abs(Wf), axis=1, keepdims=True)      # reduce over INPUT dim
    scale = np.where(amax == 0.0, 1.0, amax / qmax)
    q = np.clip(np.rint(Wf / scale), -qmax, qmax).astype(np.int8)
    return q, scale
