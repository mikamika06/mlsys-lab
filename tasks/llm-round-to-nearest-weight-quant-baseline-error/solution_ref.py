import numpy as np

def round_to_nearest(W: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Symmetric round‑to‑nearest quantization.
    """
    max_val = 0.0
    for val in W.flat:
        v = float(val)
        abs_v = v if v >= 0.0 else -v
        if abs_v > max_val:
            max_val = abs_v

    scale = max_val / ((2 ** (num_bits - 1)) - 1)
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    dtype = np.int8 if num_bits <= 8 else np.int16
    Q = np.empty(W.shape, dtype=dtype)

    for i, val in enumerate(W.flat):
        q = round(float(val) / scale)
        if q < qmin:
            q = qmin
        elif q > qmax:
            q = qmax
        Q.flat[i] = q

    return Q
