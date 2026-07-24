def linear_rope(positions, dim, L_train, L_target):
    import numpy as np
    positions = np.asarray(positions, dtype=np.float64)
    scale = L_target / L_train
    p_scaled = positions * scale
    freq = 1.0 / (10000 ** (np.arange(dim) / dim))
    theta = np.outer(p_scaled, freq)
    return np.sin(theta)
