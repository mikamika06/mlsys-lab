import numpy as np


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def log_softmax_vjp(x, g):
    """Vector-Jacobian product of y = log_softmax(x, axis=-1).

    Given the upstream gradient `g` (dLoss/dy, same shape as x), returns
    dLoss/dx = g - softmax(x) * sum(g, axis=-1, keepdims=True).
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    softmax = np.exp(_log_softmax(x))
    return g - softmax * np.sum(g, axis=-1, keepdims=True)
