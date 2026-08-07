import numpy as np

def generate_fixtures():
    np.random.seed(42)
    weights = [np.random.randn(32, 64).astype(np.float32) for _ in range(3)]
    return weights
