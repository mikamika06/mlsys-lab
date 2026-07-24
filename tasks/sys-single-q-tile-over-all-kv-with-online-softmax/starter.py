import numpy as np


def flash_forward_single_q_tile(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 kv_block_size: int) -> np.ndarray:
    """
    Fixed single query tile Q (do not sub-tile it); stream K, V in tiles of
    kv_block_size rows, maintaining running online-softmax stats (m, l, O)
    across tiles. Must not materialize the dense (n_q, n_kv) score matrix
    in one shot.

    Q: (n_q, d). K, V: (n_kv, d). Returns O: (n_q, d).
    """
    raise NotImplementedError('your code here')
