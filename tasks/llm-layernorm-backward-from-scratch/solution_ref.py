import numpy as np

def compute_dx(dy: np.ndarray,
               x: np.ndarray,
               gamma: np.ndarray,
               beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """
    Analytic gradient of LayerNorm with respect to the input `x`.

    Parameters
    ----------
    dy : ndarray, shape (N, D)
        Gradient of a loss w.r.t. the LayerNorm output.
    x : ndarray, shape (N, D)
        Original input to the forward pass.
    gamma : ndarray, shape (D,)
        Scale parameter used in the forward pass.
    beta : ndarray, shape (D,)
        Shift parameter used in the forward pass (unused in gradient).
    eps : float
        Small constant added to variance for numerical stability.

    Returns
    -------
    dx : ndarray, shape (N, D)
        Gradient of the loss w.r.t. `x`.
    """
    # Forward statistics
    mu = x.mean(axis=1, keepdims=True)                 # (N, 1)
    var = x.var(axis=1, keepdims=True) + eps           # (N, 1)
    std_inv = 1.0 / np.sqrt(var)                       # (N, 1)

    # Normalized input
    x_hat = (x - mu) * std_inv                         # (N, D)

    # Element‑wise product with gamma
    dy_gamma = dy * gamma                              # (N, D)

    # Summations needed for the analytic formula
    sum_dy = np.sum(dy_gamma, axis=1, keepdims=True)   # (N, 1)
    sum_dxhat = np.sum(dy_gamma * x_hat, axis=1, keepdims=True)  # (N, 1)

    # Final gradient expression
    dx = (dy_gamma - sum_dy / x.shape[1] - x_hat * sum_dxhat / x.shape[1]) * std_inv

    return dx
