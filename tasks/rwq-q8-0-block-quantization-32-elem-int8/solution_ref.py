import numpy as np
from typing import Tuple

def q8_0_quantize(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize a 1‑D array using the Q8_0 format (32‑element blocks, int8).

    Parameters
    ----------
    arr : np.ndarray
        1‑D array of arbitrary length and dtype convertible to float64.

    Returns
    -------
    codes : np.ndarray
        int8 array of the same shape as `arr` containing the quantized codes.
    dequant : np.ndarray
        float64 array of the same shape as `arr` containing the reconstructed values.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    block_size = 32
    codes = np.empty(n, dtype=np.int8)
    deq = np.empty(n, dtype=np.float64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = arr[start:end]
        amax = np.max(np.abs(block))
        if amax == 0:
            d = 1.0
        else:
            d = amax / 127.0
        q = np.round(block / d).astype(np.int8)
        codes[start:end] = q
        deq[start:end] = q.astype(np.float64) * d

    return codes, deq
