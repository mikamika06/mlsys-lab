import math
import numpy as np


def linear_rope(positions, dim, L_train, L_target):
    positions = np.asarray(positions, dtype=np.float64)
    scale = L_target / L_train
    n = len(positions)

    freq = [0.0] * dim
    for j in range(dim):
        freq[j] = 1.0 / (10000.0 ** (j / dim))

    out = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        p_val = positions[i] * scale
        for j in range(dim):
            out[i, j] = math.sin(p_val * freq[j])

    return out
