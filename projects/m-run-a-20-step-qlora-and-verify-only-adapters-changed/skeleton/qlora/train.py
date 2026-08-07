import numpy as np


def train_20_steps(layer, X, target, lr=0.01):
    """
    Run 20 steps of gradient descent using MSE loss.
    Loss is mean((Y - target)^2).
    grad_output = 2.0 * (Y - target) / Y.size
    Return the list of losses (length 20).
    """
    raise NotImplementedError
