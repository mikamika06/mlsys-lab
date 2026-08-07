import numpy as np


def online_softmax_attention_forward(Q, K, V, sm_scale):
    B, H, N, D = Q.shape
    O = np.zeros_like(Q, dtype=np.float64)
    Q64 = Q.astype(np.float64)
    K64 = K.astype(np.float64)
    V64 = V.astype(np.float64)

    for b in range(B):
        for h in range(H):
            q = Q64[b, h]
            k = K64[b, h]
            v = V64[b, h]

            m = np.full((N, 1), -np.inf, dtype=np.float64)
            l = np.zeros((N, 1), dtype=np.float64)
            acc = np.zeros((N, D), dtype=np.float64)

            step = 32
            for j_start in range(0, N, step):
                j_end = min(j_start + step, N)
                k_block = k[j_start:j_end]
                v_block = v[j_start:j_end]

                s_block = np.dot(q, k_block.T) * sm_scale
                m_block = np.max(s_block, axis=-1, keepdims=True)

                m_new = np.maximum(m, m_block)
                alpha = np.exp(m - m_new)
                beta = np.exp(m_block - m_new)

                p_block = np.exp(s_block - m_block)
                l_block = np.sum(p_block, axis=-1, keepdims=True)

                l = l * alpha + l_block * beta
                acc = acc * alpha + np.dot(p_block, v_block) * beta
                m = m_new

            O[b, h] = acc / l

    return O.astype(Q.dtype)
