import math
import numpy as np

def per_neuron_importance(up_proj: np.ndarray,
                          down_proj: np.ndarray) -> np.ndarray:
    """
    Compute the group L2 norm importance for each hidden neuron.

    Parameters
    ----------
    up_proj : np.ndarray, shape (h, d_in)
        Incoming weight matrix; rows correspond to neurons.
    down_proj : np.ndarray, shape (d_out, h)
        Outgoing weight matrix; columns correspond to neurons.

    Returns
    -------
    importance : np.ndarray, shape (h,)
        L2 norm of the concatenated incoming and outgoing weights for each neuron.
    """
    h = up_proj.shape[0]
    d_in = up_proj.shape[1]
    d_out = down_proj.shape[0]
    
    importance = np.empty(h, dtype=np.float64)
    for i in range(h):
        row_sum_sq = 0.0
        for j in range(d_in):
            val = up_proj[i, j]
            row_sum_sq += val * val
        col_sum_sq = 0.0
        for k in range(d_out):
            val = down_proj[k, i]
            col_sum_sq += val * val
        importance[i] = math.sqrt(row_sum_sq + col_sum_sq)
    return importance
