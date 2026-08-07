import numpy as np


def get_test_values():
    rng = np.random.default_rng(42)
    vals = rng.integers(-1, 2, size=253, dtype=np.int8)
    return vals


def get_codebook_params():
    cb = [0.1, -0.3, 0.4, -0.1, 0.2, -0.5, 0.3, -0.2]
    iw = [1.5, 5.0, 0.2, 1.1, 3.2, 0.5, 2.0, 1.0]
    return cb, iw
