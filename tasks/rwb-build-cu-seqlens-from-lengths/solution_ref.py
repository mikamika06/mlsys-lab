import numpy as np

def build_cu_seqlens(lengths: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative‑sum offset array (cu_seqlens) for a packed batch.

    Parameters
    ----------
    lengths : np.ndarray
        1‑D integer array of sequence lengths, dtype int32.

    Returns
    -------
    np.ndarray
        1‑D int32 array of length len(lengths)+1 containing the offsets.
    """
    lengths = np.asarray(lengths, dtype=np.int32)
    n = len(lengths)
    out = np.zeros(n + 1, dtype=np.int32)
    current_sum = 0
    for i in range(n):
        current_sum += lengths[i]
        out[i + 1] = current_sum
    return out
