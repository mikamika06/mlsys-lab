import numpy as np


def get_test_data():
    np.random.seed(123)
    base = [np.random.randn(8, 8)]
    a = [np.random.randn(2, 8)]
    b = [np.random.randn(8, 2)]
    prompts = [np.random.randn(8, 1) for _ in range(10)]
    return base, a, b, prompts
