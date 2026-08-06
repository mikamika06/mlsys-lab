import numpy as np


def get_fixtures():
    np.random.seed(123)
    weights = np.random.randn(16, 64)
    activations = np.random.randn(200, 64)
    return weights, activations
