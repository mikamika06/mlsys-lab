import numpy as np


def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int = 32) -> np.ndarray:
    """FlashAttention-2-style forward pass: tiled online softmax.

    Sweeps Q in row blocks and K/V in column blocks, maintaining a running
    max and running normalizer per query row so the result exactly matches
    dense softmax attention without ever materializing an (N, N) matrix.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d_k = Q.shape
    d_v = V.shape[1]
    scale = 1.0 / np.sqrt(d_k)

    O = np.zeros((n, d_v), dtype=np.float64)
    n_blocks = (n + block_size - 1) // block_size

    for i in range(n_blocks):
        q_lo, q_hi = i * block_size, min(i * block_size + block_size, n)
        br = q_hi - q_lo
        Q_block = Q[q_lo:q_hi]

        m = np.full((br,), -np.inf)
        l = np.zeros((br,), dtype=np.float64)
        O_block = np.zeros((br, d_v), dtype=np.float64)

        for j in range(n_blocks):
            kv_lo, kv_hi = j * block_size, min(j * block_size + block_size, n)
            K_block = K[kv_lo:kv_hi]
            V_block = V[kv_lo:kv_hi]

            S = Q_block @ K_block.T * scale
            row_max = S.max(axis=1)
            m_new = np.maximum(m, row_max)

            P = np.exp(S - m_new[:, None])
            l_cur = P.sum(axis=1)
            rescale = np.exp(m - m_new)

            O_block = O_block * rescale[:, None] + P @ V_block
            l = l * rescale + l_cur
            m = m_new

        O[q_lo:q_hi] = O_block / l[:, None]

    return O
