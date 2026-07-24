import numpy as np


def matmul_vjp(A: np.ndarray, B: np.ndarray, G: np.ndarray):
    dA = G @ B.T
    dB = A.T @ G
    return dA, dB
