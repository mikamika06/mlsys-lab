import numpy as np

def get_test_data():
    np.random.seed(42)
    weights = np.random.randn(16, 16)
    quantized = weights + np.random.randn(16, 16) * 0.1
    matrix = np.eye(16)
    grid = np.linspace(-2.0, 2.0, 9)
    return weights, quantized, matrix, grid
