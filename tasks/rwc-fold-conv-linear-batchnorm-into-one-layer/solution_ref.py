import math


def fold_bn_into_linear(
    W: list[list[float]],
    b: list[float],
    gamma: list[float],
    beta: list[float],
    running_mean: list[float],
    running_var: list[float],
    eps: float,
) -> tuple[list[list[float]], list[float]]:
    out_f = len(W)
    in_f = len(W[0]) if out_f > 0 else 0

    scale = [0.0] * out_f
    for i in range(out_f):
        scale[i] = gamma[i] / math.sqrt(running_var[i] + eps)

    W_folded = [[0.0] * in_f for _ in range(out_f)]
    for i in range(out_f):
        for j in range(in_f):
            W_folded[i][j] = W[i][j] * scale[i]

    b_folded = [0.0] * out_f
    for i in range(out_f):
        b_folded[i] = scale[i] * (b[i] - running_mean[i]) + beta[i]

    return W_folded, b_folded
