import numpy as np


def generate_test_data():
    np.random.seed(42)
    w = np.random.randn(32, 64).astype(np.float32)
    return w
