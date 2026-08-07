import numpy as np


def get_test_data():
    np.random.seed(1337)
    weights = np.random.randn(32, 32) * 0.4
    scale = 0.05
    zero_point = 0.0
    return weights, scale, zero_point


def get_tiny_model():
    np.random.seed(1337)
    return {
        "weights": [np.random.randn(16, 16) * 0.2, np.random.randn(16, 16) * 0.3],
        "scale": 0.1,
        "zero_point": 0.0
    }
