import numpy as np


def simulate_nf4_cycles(tensor, steps):
    levels = np.array([-1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0, 0.0795800, 0.1609302, 0.2479011, 0.3478914, 0.4679901, 0.6106249, 0.8160759, 1.0], dtype=np.float32)
    current = tensor.copy()
    for _ in range(steps):
        flat = current.ravel()
        indices = np.abs(flat[:, None] - levels[None, :]).argmin(axis=1)
        quantized = levels[indices].reshape(current.shape)
        current = quantized
    error = np.mean(np.square(current - tensor))
    return current, error
