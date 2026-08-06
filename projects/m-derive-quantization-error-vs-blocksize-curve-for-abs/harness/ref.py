import numpy as np

def generate_tensor():
    np.random.seed(42)
    return np.random.randn(64, 64).astype(np.float32) * 5.0
