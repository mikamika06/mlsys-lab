import numpy as np


def flash_forward_single_q_tile(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 kv_block_size: int) -> np.ndarray:
    """
    Fixed single query tile Q, streamed over K/V in tiles of kv_block_size,
    maintaining online-softmax running stats (m, l, O).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n_q, d = Q.shape
    n_kv = K.shape[0]
    scale = 1.0 / np.sqrt(d)

    m = np.full((n_q,), -np.inf)
    l = np.zeros((n_q,), dtype=np.float64)
    O = np.zeros((n_q, V.shape[1]), dtype=np.float64)

    n_blocks = (n_kv + kv_block_size - 1) // kv_block_size
    for j in range(n_blocks):
        lo = j * kv_block_size
        hi = min(lo + kv_block_size, n_kv)
        K_block = K[lo:hi]
        V_block = V[lo:hi]

        S = (Q @ K_block.T) * scale
        row_max = S.max(axis=1)
        m_new = np.maximum(m, row_max)

        P = np.exp(S - m_new[:, None])
        l_cur = P.sum(axis=1)
        rescale = np.exp(m - m_new)

        O = O * rescale[:, None] + P @ V_block
        l = l * rescale + l_cur
        m = m_new

    return O / l[:, None]
