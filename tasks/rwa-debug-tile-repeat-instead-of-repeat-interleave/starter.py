import numpy as np


def expand_kv_heads(kv: np.ndarray, n_query_heads: int) -> np.ndarray:
    # TODO: replace tile with head-wise interleaving.
    # This repeats the entire KV head list and sends query heads to the wrong
    # key/value groups when multiple KV heads are present.
    kv = np.asarray(kv, dtype=np.float64)
    repeat = n_query_heads // kv.shape[1]
    return np.tile(kv, (1, repeat, 1, 1))
