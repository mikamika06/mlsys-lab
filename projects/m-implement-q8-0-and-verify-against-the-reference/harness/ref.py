import numpy as np

def generate_test_weights():
    rng = np.random.default_rng(42)
    return rng.standard_normal((64, 64)).astype(np.float32)
