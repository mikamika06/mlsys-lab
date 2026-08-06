import numpy as np


def quantize_per_channel(W: np.ndarray, n_bits: int = 8):
    """Symmetric per-output-channel (row-wise) integer quantization of ``W``.

    Returns ``(q, scale)`` with int8 codes and a ``(C_out, 1)`` scale.
    """
    qmax = 2 ** (n_bits - 1) - 1
    n_out, n_in = W.shape
    q = np.empty((n_out, n_in), dtype=np.int8)
    scale = np.empty((n_out, 1), dtype=np.float64)

    for i in range(n_out):
        amax = 0.0
        for j in range(n_in):
            val = float(W[i, j])
            abs_val = abs(val)
            if abs_val > amax:
                amax = abs_val

        if amax == 0.0:
            s = 1.0
        else:
            s = amax / qmax
        scale[i, 0] = s

        for j in range(n_in):
            scaled = float(W[i, j]) / s
            r = round(scaled)
            if r > qmax:
                r = qmax
            elif r < -qmax:
                r = -qmax
            q[i, j] = r

    return q, scale
