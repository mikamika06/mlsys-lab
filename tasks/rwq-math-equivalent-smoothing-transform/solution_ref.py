import numpy as np

def smoothing_transform(X: np.ndarray, W: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the math‑equivalent smoothing transform.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input matrix.
    W : np.ndarray of shape (d, m)
        Weight matrix.
    s : np.ndarray of shape (d,)
        Scaling vector; must contain no zeros.

    Returns
    -------
    Xp : np.ndarray of shape (n, d)
        Scaled input matrix.
    Wp : np.ndarray of shape (d, m)
        Scaled weight matrix.
    """
    # Ensure float64 for deterministic behaviour
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    # Compute X' = X * diag(1/s)  (element‑wise division by s)
    Xp = X / s

    # Compute W' = diag(s) @ W
    Wp = np.diag(s) @ W

    return Xp, Wp
