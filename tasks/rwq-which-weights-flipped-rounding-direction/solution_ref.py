import numpy as np

def classify_rounding(W: np.ndarray, V: np.ndarray, s: float) -> np.ndarray:
    """
    Classify each weight according to how its rounded value changes when a
    correction vector is added.

    Parameters
    ----------
    W : np.ndarray
        Original weights.
    V : np.ndarray
        Correction vector of the same shape as `W`.
    s : float
        Positive scaling factor used before rounding.

    Returns
    -------
    np.ndarray
        Integer array with values -1 (rounded down), 0 (no change),
        or +1 (rounded up).
    """
    out = np.empty(W.shape, dtype=np.int8)
    it = np.nditer([W, V, out], flags=['external_loop', 'buffered'], op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for w_chunk, v_chunk, out_chunk in it:
        for i in range(w_chunk.shape[0]):
            r0 = round(w_chunk[i] / s)
            r1 = round((w_chunk[i] + v_chunk[i]) / s)
            if r1 > r0:
                out_chunk[i] = 1
            elif r1 < r0:
                out_chunk[i] = -1
            else:
                out_chunk[i] = 0
    return out
