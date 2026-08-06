import math
import numpy as np


def normalized_hadamard(n: int) -> np.ndarray:
    h = [[1.0]]
    while len(h) < n:
        new_h = []
        for row in h:
            new_h.append(row + row)
        for row in h:
            neg_row = []
            for val in row:
                neg_row.append(-val)
            new_h.append(row + neg_row)
        h = new_h

    scale = math.sqrt(n)
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            result[i, j] = h[i][j] / scale

    return result
