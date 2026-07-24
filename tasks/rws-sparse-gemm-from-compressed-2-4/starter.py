import numpy as np


def compressed_matmul(values: np.ndarray, idx: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    values: (d_out, d_in//2) nonzero weight values, 2 per group of 4
    columns. idx: same shape, each entry in {0,1,2,3} = the value's
    column position within its group of 4. X: (d_in, n) activations.
    Reconstruct the dense (d_out, d_in) weight matrix from the compressed
    layout and return W @ X. See task.md.
    """
    raise NotImplementedError('your code here')
