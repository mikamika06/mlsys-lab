import numpy as np


def get_test_fixtures():
    np.random.seed(42)
    activations = [np.random.randn(16, 64).astype(np.float32) * 3.5 for _ in range(3)]
    weight = np.random.randn(32, 64).astype(np.float32) * 1.2
    return activations, weight
