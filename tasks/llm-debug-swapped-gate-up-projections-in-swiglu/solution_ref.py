import math
import numpy as np


def _silu(x):
    """Sigmoid‑linear unit."""
    return x / (1.0 + math.exp(-x))


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
    n = X.shape[0]
    d_in = X.shape[1]
    d_out = W_gate.shape[1]

    Y = np.zeros((n, d_out))

    for i in range(n):
        for j in range(d_out):
            gate_val = 0.0
            for k in range(d_in):
                gate_val += float(X[i, k]) * float(W_gate[k, j])
            if b_gate is not None:
                gate_val += float(b_gate[j])

            up_val = 0.0
            for k in range(d_in):
                up_val += float(X[i, k]) * float(W_up[k, j])
            if b_up is not None:
                up_val += float(b_up[j])

            Y[i, j] = gate_val * _silu(up_val)

    return Y
