import numpy as np


def generate_gelu_inputs():
    np.random.seed(42)
    return np.random.randn(64, 64).astype(np.float32)


def gelu_reference(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def generate_matmul_inputs():
    np.random.seed(42)
    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)
    return a, b


def matmul_reference(a, b):
    return np.matmul(a, b)
