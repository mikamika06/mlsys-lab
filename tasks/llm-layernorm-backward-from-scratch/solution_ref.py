import math

def compute_dx(dy: list[list[float]],
               x: list[list[float]],
               gamma: list[float],
               beta: list[float],
               eps: float = 1e-5) -> list[list[float]]:
    """
    Analytic gradient of LayerNorm with respect to the input `x`.

    Parameters
    ----------
    dy : list[list[float]], shape (N, D)
        Gradient of a loss w.r.t. the LayerNorm output.
    x : list[list[float]], shape (N, D)
        Original input to the forward pass.
    gamma : list[float], shape (D,)
        Scale parameter used in the forward pass.
    beta : list[float], shape (D,)
        Shift parameter used in the forward pass (unused in gradient).
    eps : float
        Small constant added to variance for numerical stability.

    Returns
    -------
    dx : list[list[float]], shape (N, D)
        Gradient of the loss w.r.t. `x`.
    """
    N = len(x)
    D = len(x[0]) if N > 0 else 0

    dx = [[0.0] * D for _ in range(N)]

    for i in range(N):
        acc_mean = 0.0
        for j in range(D):
            acc_mean += x[i][j]
        m = acc_mean / D

        acc_var = 0.0
        for j in range(D):
            diff = x[i][j] - m
            acc_var += diff * diff
        v = (acc_var / D) + eps

        s_inv = 1.0 / math.sqrt(v)

        dy_gamma = [0.0] * D
        x_hat = [0.0] * D
        for j in range(D):
            xh = (x[i][j] - m) * s_inv
            x_hat[j] = xh
            dy_gamma[j] = dy[i][j] * gamma[j]

        acc_sum_dy = 0.0
        for j in range(D):
            acc_sum_dy += dy_gamma[j]

        acc_sum_dxhat = 0.0
        for j in range(D):
            acc_sum_dxhat += dy_gamma[j] * x_hat[j]

        s_dy = acc_sum_dy
        s_dxhat = acc_sum_dxhat

        for j in range(D):
            val = (dy_gamma[j] - s_dy / D - x_hat[j] * s_dxhat / D) * s_inv
            dx[i][j] = val

    return dx
