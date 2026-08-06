import numpy as np
from typing import Tuple

def per_token_int8_quant(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per‑token int8 quantisation.

    Parameters
    ----------
    A : np.ndarray
        2-D array of shape (n, d).

    Returns
    -------
    codes : np.ndarray
        Quantised integer codes, dtype=np.int8.
    scales : np.ndarray
        Per‑row scale factors, dtype=float64.
    """
    n, d = A.shape
    scales = np.empty(n, dtype=np.float64)
    codes = np.empty((n, d), dtype=np.int8)

    for i in range(n):
        row = A[i]
        max_abs = 0.0
        for j in range(d):
            val = row[j]
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs:
                max_abs = abs_val
        
        scale = 1.0 if max_abs == 0.0 else max_abs / 127.0
        scales[i] = scale

        for j in range(d):
            val = row[j] / scale
            rounded = round(val)
            if rounded < -128.0:
                clipped = -128
            elif rounded > 127.0:
                clipped = 127
            else:
                clipped = int(rounded)
            codes[i, j] = clipped

    return codes, scales
