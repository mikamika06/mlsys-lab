import numpy as np


def get_test_weights():
    rng = np.random.default.rng(1337)
    return [rng.uniform(-3.0, 3.0, 256).astype(np.float32) for _ in range(5)]
