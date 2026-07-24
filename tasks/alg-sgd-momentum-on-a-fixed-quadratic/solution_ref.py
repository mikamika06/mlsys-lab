import numpy as np

def sgd_momentum_quadratic(
    A: np.ndarray,
    b: np.ndarray,
    init_x: np.ndarray,
    lr: float,
    momentum: float,
    T: int
) -> np.ndarray:
    """
    Heavy‑ball (SGD with momentum) on a quadratic objective.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Symmetric positive‑definite matrix.
    b : ndarray of shape (n,)
        Linear term vector.
    init_x : ndarray of shape (n,)
        Starting point.
    lr : float
        Learning rate η.
    momentum : float
        Momentum coefficient β ∈ [0, 1).
    T : int
        Number of iterations.

    Returns
    -------
    x_T : ndarray of shape (n,)
        Final iterate after T updates.
    """
    x = init_x.astype(np.float64)
    v = np.zeros_like(x)
    for _ in range(T):
        grad = A @ x - b
        v = momentum * v - lr * grad
        x = x + v
    return x
