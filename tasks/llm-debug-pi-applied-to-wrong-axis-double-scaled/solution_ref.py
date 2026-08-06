import numpy as np

def rope_pi(seq_len, dim, L_train, L_new):
    """RoPE with Position Interpolation: scale only the positions."""
    pos = np.arange(seq_len, dtype=np.float64)
    scale = L_train / L_new
    pos_scaled = pos * scale                              # scale positions only
    k = np.arange(dim // 2, dtype=np.float64)
    theta = 1.0 / (10000.0 ** (2.0 * k / dim))          # no extra scale
    angles = np.outer(pos_scaled, theta)
    return np.cos(angles), np.sin(angles)
