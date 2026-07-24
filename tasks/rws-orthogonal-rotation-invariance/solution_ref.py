import numpy as np


def rotate_and_matvec(W: np.ndarray, x: np.ndarray, Q: np.ndarray):
    """
    Rotate a linear layer by an orthogonal matrix Q and show the output
    is unchanged: W_rot = W @ Q, x_rot = Q.T @ x, y = W_rot @ x_rot (which
    equals W @ x since Q Q.T = I). Returns (W_rot, x_rot, y).
    """
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    W_rot = W @ Q
    x_rot = Q.T @ x
    y = W_rot @ x_rot
    return W_rot, x_rot, y
