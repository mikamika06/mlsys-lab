import math
import numpy as np


def det_from_lu(A: np.ndarray) -> float:
    M = np.asarray(A, dtype=np.float64).copy()
    n = M.shape[0]
    swaps = 0

    for k in range(n):
        max_val = -1.0
        pivot = k
        for i in range(k, n):
            val = math.fabs(float(M[i, k]))
            if val > max_val:
                max_val = val
                pivot = i

        if pivot != k:
            for j in range(n):
                tmp = float(M[k, j])
                M[k, j] = M[pivot, j]
                M[pivot, j] = tmp
            swaps += 1

        if M[k, k] == 0:
            return 0.0

        for i in range(k + 1, n):
            factor = float(M[i, k]) / float(M[k, k])
            for j in range(k + 1, n):
                M[i, j] -= factor * float(M[k, j])

    sign = -1.0 if swaps % 2 else 1.0
    prod = 1.0
    for i in range(n):
        prod *= float(M[i, i])

    return float(sign * prod)
