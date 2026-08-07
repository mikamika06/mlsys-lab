import numpy as np


def reconstruct_kept_mask(compressed_dump, original_length):
    indices = compressed_dump.get("indices")
    if indices is not None:
        mask = np.zeros(original_length, dtype=bool)
        mask[indices] = True
        return mask
    timestamps = compressed_dump.get("timestamps")
    if timestamps is not None:
        threshold = compressed_dump.get("threshold", 0.0)
        return np.array(timestamps >= threshold, dtype=bool)
    return np.ones(original_length, dtype=bool)
