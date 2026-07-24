import numpy as np

def tiled_flash_attention_forward(Q, K, V, block_size=64):
    """Block-tiled flash-attention forward pass (online softmax).

    Never materialises the full n x n score matrix.
    """
    n = Q.shape[0]
    d_k = Q.shape[1]
    d_v = V.shape[1]
    scale = 1.0 / np.sqrt(d_k)

    O = np.zeros((n, d_v), dtype=np.float64)

    n_blocks = (n + block_size - 1) // block_size

    for i in range(n_blocks):
        q_lo = i * block_size
        q_hi = min(q_lo + block_size, n)
        br = q_hi - q_lo

        Q_block = Q[q_lo:q_hi, :]          # (br, d_k)

        # running online-softmax state
        m = np.full((br,), -np.inf)        # running row-max
        l = np.zeros((br,), dtype=np.float64)  # running row-sum
        O_block = np.zeros((br, d_v), dtype=np.float64)

        for j in range(n_blocks):
            kv_lo = j * block_size
            kv_hi = min(kv_lo + block_size, n)

            K_block = K[kv_lo:kv_hi, :]    # (bc, d_k)
            V_block = V[kv_lo:kv_hi, :]    # (bc, d_v)

            S = Q_block @ K_block.T * scale # (br, bc)

            row_max = S.max(axis=1)        # (br,)
            m_new = np.maximum(m, row_max)

            P = np.exp(S - m_new[:, None])  # (br, bc)
            l_cur = P.sum(axis=1)           # (br,)

            rescale = np.exp(m - m_new)     # (br,)
            O_block = O_block * rescale[:, None] + P @ V_block
            l = l * rescale + l_cur

            m = m_new

        O[q_lo:q_hi, :] = O_block / l[:, None]

    return O
