import math
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
    N, D = x.shape
    
    mu = np.empty((N, 1), dtype=x.dtype)
    var = np.empty((N, 1), dtype=x.dtype)
    std_inv = np.empty((N, 1), dtype=x.dtype)
    x_hat = np.empty((N, D), dtype=x.dtype)
    dy_gamma = np.empty((N, D), dtype=x.dtype)
    sum_dy = np.empty((N, 1), dtype=x.dtype)
    sum_dxhat = np.empty((N, 1), dtype=x.dtype)
    dx = np.empty((N, D), dtype=x.dtype)

    for i in range(N):
        acc_mean = 0.0
        for j in range(D):
            acc_mean += x[i, j]
        m = acc_mean / D
        mu[i, 0] = m

        acc_var = 0.0
        for j in range(D):
            diff = x[i, j] - m
            acc_var += diff * diff
        v = (acc_var / D) + eps
        var[i, 0] = v

        s_inv = 1.0 / math.sqrt(v)
        std_inv[i, 0] = s_inv

        for j in range(D):
            xh = (x[i, j] - m) * s_inv
            x_hat[i, j] = xh
            dg = dy[i, j] * gamma[j]
            dy_gamma[i, j] = dg

        acc_sum_dy = 0.0
        for j in range(D):
            acc_sum_dy += dy_gamma[i, j]
        sum_dy[i, 0] = acc_sum_dy

        acc_sum_dxhat = 0.0
        for j in range(D):
            acc_sum_dxhat += dy_gamma[i, j] * x_hat[i, j]
        sum_dxhat[i, 0] = acc_sum_dxhat

        s_dy = acc_sum_dy
        s_dxhat = acc_sum_dxhat

        for j in range(D):
            val = (dy_gamma[i, j] - s_dy / D - x_hat[i, j] * s_dxhat / D) * s_inv
            dx[i, j] = val

    return dx
