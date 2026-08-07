import math

def online_softmax_weighted_sum(scores: list[float], V: list[list[float]], block_size: int) -> list[float]:
    """softmax(scores) @ V, computed one block at a time via the online-softmax
    running (m, l, o) update -- never calling exp on the full-length score
    vector. See task.md for the update rule.
    """
    raise NotImplementedError('your code here')
