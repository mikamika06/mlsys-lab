import math

def fold_bn_into_linear(W: list[list[float]], b: list[float], gamma: list[float], beta: list[float], running_mean: list[float], running_var: list[float], eps: float) -> tuple[list[list[float]], list[float]]:
    """Fold a frozen BatchNorm into the preceding Linear layer.

    W: (out_features, in_features) float64. b: (out_features,) float64.
    gamma, beta, running_mean, running_var: (out_features,) float64
        BatchNorm parameters. eps: float.

    scale = gamma / sqrt(running_var + eps)
    W_folded = scale[:, None] * W
    b_folded = scale * (b - running_mean) + beta

    Returns (W_folded, b_folded) such that W_folded @ x + b_folded
    equals BN(W @ x + b) for every x.
    """
    raise NotImplementedError('your code here')
