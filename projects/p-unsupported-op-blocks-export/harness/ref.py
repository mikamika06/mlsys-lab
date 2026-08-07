import numpy as np

def get_test_data():
    np.random.seed(123)
    return np.random.randn(32, 32).astype(np.float32)

def reference_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
