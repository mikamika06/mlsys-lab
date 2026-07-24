import numpy as np


def transpose_view(A: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    return np.transpose(A, axes=axes)
