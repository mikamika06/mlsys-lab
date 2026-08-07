import numpy as np


def get_model():
    np.random.seed(42)
    W = np.random.randn(1000, 1000)
    b = np.zeros(1000)
    return W, b


def get_generator():
    def generator(steps):
        np.random.seed(100 + steps)
        return [(np.random.randn(500, 1000), np.random.randn(500, 1000)) for _ in range(steps)]
    return generator


def get_fixed_batches(steps):
    np.random.seed(42)
    return [(np.random.randn(200, 50), np.random.randn(200, 50)) for _ in range(steps)]
