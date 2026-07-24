import numpy as np
from typing import Tuple

def per_token_int8_quant(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per‑token int8 quantisation (broken implementation).

Parameters
----------
A : np.ndarray
    2-D array of shape (n, d).

Returns
-------
codes : np.ndarray
    Quantised integer codes, dtype=np.int8.
scales : np.ndarray
    Per‑row scale factors, dtype=float64."""
    raise NotImplementedError('your code here')
