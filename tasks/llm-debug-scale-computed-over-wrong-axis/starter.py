import numpy as np


def quantize_per_channel(W: np.ndarray, n_bits: int = 8):
    """Symmetric per-output-channel (row-wise) integer quantization of ``W``.

    BUG REPORT: every output channel (row of ``W``) is supposed to get its own
    scale, but after a round trip the quiet channels come back as garbage while
    the loud ones look fine. Find the defect and fix it.

    Keep the return contract: ``(q, scale)`` with ``q`` of dtype ``np.int8`` and
    a ``scale`` that broadcasts against ``W`` so ``q * scale`` reconstructs it.
    """
    qmax = 2 ** (n_bits - 1) - 1
    Wf = np.asarray(W, dtype=np.float64)
    amax = np.max(np.abs(Wf), axis=0, keepdims=True)
    scale = np.where(amax == 0.0, 1.0, amax / qmax)
    q = np.clip(np.rint(Wf / scale), -qmax, qmax).astype(np.int8)
    return q, scale
