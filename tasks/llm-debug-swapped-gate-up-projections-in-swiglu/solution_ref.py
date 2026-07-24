import numpy as np

def _silu(x):
    """Sigmoid‑linear unit."""
    return x / (1.0 + np.exp(-x))

def swiglu(
    X: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    b_gate: np.ndarray | None = None,
    b_up: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the SwiGLU activation.

    Parameters
    ----------
    X : (n, d_in) array of inputs.
    W_gate, W_up : weight matrices of shape (d_in, d_out).
    b_gate, b_up : optional bias vectors of length d_out; treated as zero if None.

    Returns
    -------
    Y : (n, d_out) array of SwiGLU outputs.
    """
    gate = X @ W_gate + (b_gate if b_gate is not None else 0.0)
    up   = X @ W_up   + (b_up   if b_up   is not None else 0.0)
    return gate * _silu(up)
