import numpy as np

def fused_swiglu(x, w_up, b_up, w_gate, b_gate):
    """
    Compute (x @ w_up + b_up) * swish(x @ w_gate + b_gate)
    using a single matrix multiplication by concatenating weights and biases.
    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input batch.
    w_up : np.ndarray, shape (d, h)
        Weight matrix for the linear part.
    b_up : np.ndarray, shape (h,)
        Bias vector for the linear part.
    w_gate : np.ndarray, shape (d, h)
        Weight matrix for the gate.
    b_gate : np.ndarray, shape (h,)
        Bias vector for the gate.
    Returns
    -------
    y : np.ndarray, shape (n, h), dtype float64
        SwiGLU output.
    """
    # concatenate weights and biases
    w_concat = np.concatenate([w_up, w_gate], axis=1)   # (d, 2h)
    b_concat = np.concatenate([b_up, b_gate])           # (2h,)
    tmp      = x @ w_concat + b_concat                 # (n, 2h)

    h = w_up.shape[1]
    u = tmp[:, :h]                                      # (n, h)
    g_raw = tmp[:, h:]                                  # (n, h)
    g = g_raw / (1 + np.exp(-g_raw))                    # swish
    return u * g
