import numpy as np

def compute_clip_threshold(gradients, max_norm):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
    if total_norm > max_norm:
        return float(max_norm / total_norm)
    return 1.0
