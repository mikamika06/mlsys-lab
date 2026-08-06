import math
import itertools

def log_softmax_vjp(x: list[list[float]], g: list[list[float]]) -> list[list[float]]:
    """Vector-Jacobian product of y = log_softmax(x, axis=-1).

    Given the upstream gradient `g` (dLoss/dy, same shape as x), returns
    dLoss/dx, a float64 array the same shape as x.
    """
    raise NotImplementedError('your code here')
