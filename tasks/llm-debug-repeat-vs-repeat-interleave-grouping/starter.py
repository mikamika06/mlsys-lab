import numpy as np


def expand_kv_heads(kv: np.ndarray, num_q_heads: int) -> np.ndarray:
    # TODO: wrong grouping order. This tiles complete head sets instead of
    # repeating each key/value head for its query-head group.
    groups = num_q_heads // kv.shape[1]
    return np.tile(kv, (1, groups, 1, 1))
