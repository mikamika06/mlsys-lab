import numpy as np

def vjp_matmul(A: np.ndarray, B: np.ndarray, dY: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the vector‑Jacobian product of a matrix multiplication Y = A @ B.

    Parameters
    ----------
    A : np.ndarray
        Left operand of shape (m, k).
    B : np.ndarray
        Right operand of shape (k, n).
    dY : np.ndarray
        Upstream gradient of shape (m, n).

    Returns
    -------
    dA : np.ndarray
        Gradient with respect to A, shape (m, k).
    dB : np.ndarray
        Gradient with respect to B, shape (k, n).
    """
    # Ensure float64 for numerical stability
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    dY = np.asarray(dY, dtype=np.float64)

    dA = dY @ B.T          # (m, n) @ (n, k) -> (m, k)
    dB = A.T @ dY          # (k, m) @ (m, n) -> (k, n)
    return dA, dB
