import numpy as np


def get_test_inputs():
    return [
        np.array([1.0, -2.0, 3.0], dtype=np.float32),
        np.zeros((10, 10), dtype=np.float32),
        np.ones((4, 4), dtype=np.float32) * -1.0
    ]
