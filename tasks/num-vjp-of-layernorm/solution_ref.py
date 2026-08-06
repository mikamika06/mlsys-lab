import math


def layernorm_vjp(x: list[list[float]], grad_y: list[list[float]], eps: float = 1e-5) -> list[list[float]]:
    N = len(x)
    D = len(x[0])
    out = [[0.0] * D for _ in range(N)]

    for i in range(N):
        mean_s = 0.0
        for j in range(D):
            mean_s += x[i][j]
        mean = mean_s / D

        var_s = 0.0
        for j in range(D):
            diff = x[i][j] - mean
            var_s += diff * diff
        var = var_s / D

        inv_std = 1.0 / math.sqrt(var + eps)

        mean_grad_s = 0.0
        for j in range(D):
            mean_grad_s += grad_y[i][j]
        mean_grad = mean_grad_s / D

        mean_grad_hat_s = 0.0
        for j in range(D):
            x_hat_j = (x[i][j] - mean) * inv_std
            mean_grad_hat_s += grad_y[i][j] * x_hat_j
        mean_grad_hat = mean_grad_hat_s / D

        for j in range(D):
            x_hat_j = (x[i][j] - mean) * inv_std
            out[i][j] = inv_std * (grad_y[i][j] - mean_grad - x_hat_j * mean_grad_hat)

    return out
