import numpy as np

def mark_activations(num_layers: int, seg_lengths) -> np.ndarray:
    """
    Return an array of length `num_layers` where indices that are
    checkpoint boundaries (start of each segment) are marked with 1,
    and all other indices are 0.
    """
    if sum(seg_lengths) != num_layers:
        raise ValueError("segment lengths must sum to num_layers")
    boundaries = []
    current = 0
    for length in seg_lengths:
        boundaries.append(current)
        current += length
    labels = np.zeros(num_layers, dtype=int)
    for b in boundaries:
        labels[b] = 1
    return labels
