import numpy as np


def det_from_lu(A: np.ndarray) -> float:
    M = np.asarray(A, dtype=np.float64).copy()
    n = M.shape[0]
    swaps = 0

    for k in range(n):
        pivot = k + int(np.argmax(np.abs(M[k:, k])))
        if pivot != k:
            M[[k, pivot]] = M[[pivot, k]]
            swaps += 1

        if M[k, k] == 0:
            return 0.0

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k + 1:] -= factor * M[k, k + 1:]

    sign = -1.0 if swaps % 2 else 1.0
    return float(sign * np.prod(np.diag(M)))
