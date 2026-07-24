import numpy as np

def classify_past_tokens(k: int, w: int, pos: int) -> np.ndarray:
    """
    Return an array of length ``pos`` where each element is 0 (sink),
    1 (window), or 2 (evicted) according to the rules described in
    the context section.
    """
    labels = np.full(pos, 2, dtype=np.int64)          # start with evicted
    if pos == 0:
        return labels
    win_start = max(0, pos - w)
    labels[win_start:pos] = 1                         # window
    if k > 0:
        labels[:k] = 0                                 # sink overrides window
    return labels
