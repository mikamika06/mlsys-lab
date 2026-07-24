import numpy as np

def alibi_bias_matrix(head_slopes: np.ndarray, seq_len: int) -> np.ndarray:
    """TODO: This implementation incorrectly uses the absolute distance,
which changes the sign pattern of the bias matrix."""
    raise NotImplementedError('your code here')
