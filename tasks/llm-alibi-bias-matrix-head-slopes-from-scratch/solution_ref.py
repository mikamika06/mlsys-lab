import numpy as np

def alibi_bias_matrix(head_slopes: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Compute the ALiBi bias matrix for all heads.

    Parameters
    ----------
    head_slopes : array-like of shape (H,)
        Slope values for each attention head. Must be positive floats.
    seq_len : int
        Length of the sequence (number of positions).

    Returns
    -------
    biases : np.ndarray of shape (H, seq_len, seq_len)
        Bias matrix for each head: bias[h,i,j] = -head_slopes[h]*(i-j).
    """
    head_slopes = np.asarray(head_slopes, dtype=np.float64)
    pos = np.arange(seq_len, dtype=np.int32)
    dist = pos[:, None] - pos[None, :]          # (L,L)
    biases = -head_slopes[:, None, None] * dist[None, :, :]
    return biases
