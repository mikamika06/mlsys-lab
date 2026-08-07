import numpy as np


def get_test_inputs():
    rng = np.random.default_rng(123)
    a = rng.standard_normal((256, 1024), dtype=np.float32)
    b = rng.standard_normal((1024, 256), dtype=np.float32)
    return a, b
