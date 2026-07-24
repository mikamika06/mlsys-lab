import numpy as np


def matmul_vjp(A: np.ndarray, B: np.ndarray, G: np.ndarray):
    # TODO: fix the VJP for A. This uses B directly instead of B.T, which
    # computes the wrong chain-rule transpose for the input gradient.
    dA = G @ B
    dB = A.T @ G
    return dA, dB
