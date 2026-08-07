import numpy as np
import sparsity.core as core

def checkpoint_size(matrix: np.ndarray) -> float:
    if core.is_2_4(matrix):
        return float(matrix.size * 1.125)
    return float(matrix.size * 2)
