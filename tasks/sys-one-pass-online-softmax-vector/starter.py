import numpy as np


def online_softmax_weighted_sum(scores: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """softmax(scores) @ V, computed one block at a time via the online-softmax
    running (m, l, o) update -- never calling exp on the full-length score
    vector. See task.md for the update rule.
    """
    raise NotImplementedError('your code here')
