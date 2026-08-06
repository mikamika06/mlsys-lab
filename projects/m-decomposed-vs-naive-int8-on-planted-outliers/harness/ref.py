import numpy as np

def generate_data():
    np.random.seed(1337)
    x = np.random.randn(64, 64).astype(np.float32)
    x[5, 5] = 45.0
    x[12, 30] = -50.0
    w = np.random.randn(64, 64).astype(np.float32)
    return x, w
