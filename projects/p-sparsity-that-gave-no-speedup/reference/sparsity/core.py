import numpy as np

def is_2_4(matrix: np.ndarray) -> bool:
    if matrix.ndim != 2:
        return False
    if matrix.shape[1] % 4 != 0:
        return False
    reshaped = matrix.reshape(-1, 4)
    non_zeros = np.count_nonzero(reshaped, axis=1)
    return bool(np.all(non_zeros <= 2))
