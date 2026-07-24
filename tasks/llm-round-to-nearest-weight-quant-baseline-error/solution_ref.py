import numpy as np

def round_to_nearest(W: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Symmetric round‑to‑nearest quantization.

    Parameters
    ----------
    W : np.ndarray of float32
        Weight tensor to be quantized.
    num_bits : int
        Number of signed bits for the integer representation (e.g. 8 or 16).

    Returns
    -------
    Q : np.ndarray of int8/int16
        Quantized integer array with values in [-2^(n-1), ..., 2^(n-1)-1].
    """
    max_val = np.max(np.abs(W))
    scale = max_val / ((2**(num_bits-1)) - 1)

    # Compute raw quantized values
    Q_raw = np.round(W / scale)

    # Clip to representable range
    qmin = -(2**(num_bits-1))
    qmax = (2**(num_bits-1)) - 1
    Q_clipped = np.clip(Q_raw, qmin, qmax)

    # Cast to appropriate integer dtype
    dtype = np.int8 if num_bits <= 8 else np.int16
    return Q_clipped.astype(dtype)
