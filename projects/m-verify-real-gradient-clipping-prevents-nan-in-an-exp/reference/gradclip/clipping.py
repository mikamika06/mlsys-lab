import numpy as np


def clip_grad_norm(grads, max_norm):
    total_norm = np.sqrt(sum(np.sum(np.square(g)) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        clipped = [g * scale for g in grads]
    else:
        clipped = [g.copy() for g in grads]
    return clipped, total_norm
