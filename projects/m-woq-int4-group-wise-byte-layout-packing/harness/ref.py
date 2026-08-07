import numpy as np


def generate_test_data():
    rng = np.random.default_rng(123)
    weights = rng.standard_normal((64, 128)).astype(np.float32)
    return weights
