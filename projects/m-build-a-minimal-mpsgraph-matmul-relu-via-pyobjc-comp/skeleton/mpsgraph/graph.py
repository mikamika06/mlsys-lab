import numpy as np


class MPSGraphMatMulReLU:
    """Minimal MPSGraph MatMul + ReLU graph constructor and executor."""

    def __init__(self, shape_a: tuple[int, int], shape_b: tuple[int, int]):
        raise NotImplementedError

    def run(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compare_with_numpy(self, a: np.ndarray, b: np.ndarray) -> dict:
        raise NotImplementedError
