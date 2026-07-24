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
    # Sum of squares of incoming weights per neuron
    row_norm_sq = np.sum(up_proj**2, axis=1)
    # Sum of squares of outgoing weights per neuron
    col_norm_sq = np.sum(down_proj**2, axis=0)
    # Group L2 norm: sqrt(row^2 + col^2)
    return np.sqrt(row_norm_sq + col_norm_sq)
