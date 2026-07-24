import numpy as np

def decompose_floats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a float32 array into sign, exponent and mantissa fields.

    Parameters
    ----------
    arr : np.ndarray
        1‑D array of dtype float32.

    Returns
    -------
    signs : np.ndarray
        uint32 array containing the sign bit (0 or 1).
    exps : np.ndarray
        uint32 array containing the biased exponent (0‑255).
    mantissas : np.ndarray
        uint32 array containing the raw mantissa bits (0‑2**23-1).
    """
    bits = arr.view(np.uint32)
    signs = (bits >> 31) & 0x1
    exps = (bits >> 23) & 0xff
    mantissas = bits & 0x7fffff
    return signs.astype(np.uint32), exps.astype(np.uint32), mantissas.astype(np.uint32)
