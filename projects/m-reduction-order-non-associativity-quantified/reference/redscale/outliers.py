import numpy as np


def detect_loss_spike_ranks(loss_matrix, multiplier=3.0):
    mat = np.array(loss_matrix, dtype=np.float32)
    if mat.size == 0:
        return []
    median = np.median(mat, axis=0)
    mad = np.median(np.abs(mat - median), axis=0)
    mad = np.where(mad == 0.0, 1e-6, mad)
    scores = np.abs(mat - median) / (1.4826 * mad)
    spikes = np.argwhere(scores > multiplier)
    return [(int(r), int(c)) for r, c in spikes]
