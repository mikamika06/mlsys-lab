import numpy as np


def reconstruct_output(states):
    m = np.stack([s[0] for s in states], axis=0).astype(np.float64)
    l = np.stack([s[1] for s in states], axis=0).astype(np.float64)
    o = np.stack([s[2] for s in states], axis=0).astype(np.float64)

    global_m = np.max(m, axis=0)
    scale = np.exp(m - global_m[None, :])

    numerator = np.sum(scale[:, :, None] * o, axis=0)
    denominator = np.sum(scale * l, axis=0)

    return numerator / denominator[:, None]
