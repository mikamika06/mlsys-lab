import numpy as np


class Parameter:

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.grad = None


class DummyOptimizer:

    def __init__(self, param_groups):
        self.param_groups = param_groups


def generate_scenario():
    np.random.seed(42)
    p1 = Parameter(np.random.randn(5, 5))
    p2 = Parameter(np.random.randn(10))

    true_g1 = np.random.randn(5, 5) * 0.5
    true_g2 = np.random.randn(10) * 0.5

    scale = 1024.0
    p1.grad = true_g1 * scale
    p2.grad = true_g2 * scale

    opt = DummyOptimizer([{"params": [p1, p2]}])
    expected_grads = [[true_g1, true_g2]]
    return scale, opt, expected_grads
