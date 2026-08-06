import numpy as np
import math

def tiled_flash_attention_forward(Q, K, V, block_size=64):
    """Block-tiled flash-attention forward pass (online softmax).

    Never materialises the full n x n score matrix.
    """
    n = Q.shape[0]
    d_k = Q.shape[1]
    d_v = V.shape[1]
    scale = 1.0 / math.sqrt(d_k)

    O = np.zeros((n, d_v), dtype=np.float64)

    n_blocks = (n + block_size - 1) // block_size

    for i in range(n_blocks):
        q_lo = i * block_size
        q_hi = min(q_lo + block_size, n)
        br = q_hi - q_lo

        Q_block = Q[q_lo:q_hi, :]          # (br, d_k)

        m = np.full((br,), -float('inf'), dtype=np.float64)
        l = np.zeros((br,), dtype=np.float64)
        O_block = np.zeros((br, d_v), dtype=np.float64)

        for j in range(n_blocks):
            kv_lo = j * block_size
            kv_hi = min(kv_lo + block_size, n)
            bc = kv_hi - kv_lo

            K_block = K[kv_lo:kv_hi, :]    # (bc, d_k)
            V_block = V[kv_lo:kv_hi, :]    # (bc, d_v)

            S = np.zeros((br, bc), dtype=np.float64)
            for r in range(br):
                for c in range(bc):
                    val = 0.0
                    for k in range(d_k):
                        val += Q_block[r, k] * K_block[c, k]
                    S[r, c] = val * scale

            row_max = np.zeros((br,), dtype=np.float64)
            for r in range(br):
                mx = S[r, 0]
                for c in range(1, bc):
                    if S[r, c] > mx:
                        mx = S[r, c]
                row_max[r] = mx

            m_new = np.zeros((br,), dtype=np.float64)
            for r in range(br):
                if m[r] > row_max[r]:
                    m_new[r] = m[r]
                else:
                    m_new[r] = row_max[r]

            P = np.zeros((br, bc), dtype=np.float64)
            for r in range(br):
                for c in range(bc):
                    P[r, c] = math.exp(S[r, c] - m_new[r])

            l_cur = np.zeros((br,), dtype=np.float64)
            for r in range(br):
                s_val = 0.0
                for c in range(bc):
                    s_val += P[r, c]
                l_cur[r] = s_val

            rescale = np.zeros((br,), dtype=np.float64)
            for r in range(br):
                rescale[r] = math.exp(m[r] - m_new[r])

            for r in range(br):
                rc = rescale[r]
                l[r] = l[r] * rc + l_cur[r]
                m[r] = m_new[r]
                for v_idx in range(d_v):
                    pv_sum = 0.0
                    for c in range(bc):
                        pv_sum += P[r, c] * V_block[c, v_idx]
                    O_block[r, v_idx] = O_block[r, v_idx] * rc + pv_sum

        for r in range(br):
            inv_l = 1.0 / l[r]
            for v_idx in range(d_v):
                O[q_lo + r, v_idx] = O_block[r, v_idx] * inv_l

    return O
