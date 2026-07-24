import numpy as np

def rope_pi(seq_len, dim, L_train, L_new):
    """RoPE with Position Interpolation.

    BUG: the scale factor is applied to BOTH the positions AND the frequencies.
    Fix it so scale appears only when computing pos_scaled = pos * scale.

    Returns: (cos, sin) each of shape (seq_len, dim // 2)
    """
    pos = np.arange(seq_len, dtype=np.float64)
    scale = L_train / L_new
    pos_scaled = pos * scale
    k = np.arange(dim // 2, dtype=np.float64)
    theta = 1.0 / (10000.0 ** (2.0 * k / dim)) * scale   # BUG: extra * scale
    angles = np.outer(pos_scaled, theta)
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin
