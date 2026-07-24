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
    amax = np.max(np.abs(A), axis=1)
    # Avoid division by zero: if amax is 0, set scale to 1.0 (codes will be all zeros).
    scales = np.where(amax == 0, 1.0, amax / 127.0)
    codes = np.round(A / scales[:, None]).clip(-128, 127).astype(np.int8)
    return codes, scales
