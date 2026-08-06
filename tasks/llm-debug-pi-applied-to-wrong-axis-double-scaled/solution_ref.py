import math
import numpy as np


def rope_pi(seq_len, dim, L_train, L_new):
    """RoPE with Position Interpolation: scale only the positions."""
    scale = L_train / L_new
    half_dim = dim // 2
    cos_out = np.zeros((seq_len, half_dim), dtype=np.float64)
    sin_out = np.zeros((seq_len, half_dim), dtype=np.float64)

    theta = [0.0] * half_dim
    for j in range(half_dim):
        theta[j] = 1.0 / (10000.0 ** (2.0 * float(j) / float(dim)))

    for i in range(seq_len):
        pos_scaled = float(i) * scale
        for j in range(half_dim):
            angle = pos_scaled * theta[j]
            cos_out[i, j] = math.cos(angle)
            sin_out[i, j] = math.sin(angle)

    return cos_out, sin_out
