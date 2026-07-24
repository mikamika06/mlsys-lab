import numpy as np


def sliding_window_attention_tiled(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window: int, block_size: int) -> np.ndarray:
    """
    Sliding-window causal attention, computed tile by tile over the query
    axis. For each query tile [qs, qe), only the key/value slice
    [max(0, qs - window + 1), qe) is ever touched -- the full (n, n) mask
    or score matrix is never materialized.

    Q, K, V: (n, d) float64.
    window: query i attends to keys max(0, i-window+1) .. i.
    block_size: number of query rows processed per tile.

    Returns: (n, d) float64 attention output.
    """
    raise NotImplementedError('your code here')
