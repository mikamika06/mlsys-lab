import numpy as np


def block_split_causal_attention_forward(Q, K, V, sm_scale, block_size=32):
    B, H, N, D = Q.shape
    O = np.zeros_like(Q, dtype=np.float64)
    Q64 = Q.astype(np.float64)
    K64 = K.astype(np.float64)
    V64 = V.astype(np.float64)

    num_blocks = (N + block_size - 1) // block_size

    for b in range(B):
        for h in range(H):
            q = Q64[b, h]
            k = K64[b, h]
            v = V64[b, h]

            for i_blk in range(num_blocks):
                i_start = i_blk * block_size
                i_end = min(i_start + block_size, N)
                q_tile = q[i_start:i_end]
                n_tile = i_end - i_start

                m = np.full((n_tile, 1), -np.inf, dtype=np.float64)
                l = np.zeros((n_tile, 1), dtype=np.float64)
                acc = np.zeros((n_tile, D), dtype=np.float64)

                for j_blk in range(i_blk):
                    j_start = j_blk * block_size
                    j_end = min(j_start + block_size, N)
                    k_block = k[j_start:j_end]
                    v_block = v[j_start:j_end]

                    s_block = np.dot(q_tile, k_block.T) * sm_scale
                    m_block = np.max(s_block, axis=-1, keepdims=True)

                    m_new = np.maximum(m, m_block)
                    alpha = np.exp(m - m_new)
                    beta = np.exp(m_block - m_new)

                    p_block = np.exp(s_block - m_block)
                    l_block = np.sum(p_block, axis=-1, keepdims=True)

                    l = l * alpha + l_block * beta
                    acc = acc * alpha + np.dot(p_block, v_block) * beta
                    m = m_new

                j_start = i_blk * block_size
                j_end = min(j_start + block_size, N)
                k_block = k[j_start:j_end]
                v_block = v[j_start:j_end]

                s_block = np.dot(q_tile, k_block.T) * sm_scale

                q_idx = np.arange(i_start, i_end)[:, None]
                k_idx = np.arange(j_start, j_end)[None, :]
                causal_mask = q_idx >= k_idx

                s_block = np.where(causal_mask, s_block, -np.inf)
                m_block = np.max(s_block, axis=-1, keepdims=True)
                m_block = np.where(np.isneginf(m_block), 0.0, m_block)

                m_new = np.maximum(m, m_block)
                alpha = np.exp(m - m_new)
                beta = np.exp(m_block - m_new)

                p_block = np.where(causal_mask, np.exp(s_block - m_block), 0.0)
                l_block = np.sum(p_block, axis=-1, keepdims=True)

                l = l * alpha + l_block * beta
                acc = acc * alpha + np.dot(p_block, v_block) * beta
                m = m_new

                O[b, h, i_start:i_end] = acc / np.maximum(l, 1e-12)

    return O.astype(Q.dtype)
