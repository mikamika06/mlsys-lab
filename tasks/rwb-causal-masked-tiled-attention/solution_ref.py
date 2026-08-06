import math
import numpy as np


def tiled_causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """FlashAttention-style tiled causal self-attention with online softmax.

    Q, K, V: (n, d). block_size: tile edge length along the sequence axis.

    The (n, n) score matrix is never materialized. Instead the query and
    key/value sequences are each split into tiles of `block_size` rows, and
    for every query tile we sweep over key/value tiles LEFT TO RIGHT,
    maintaining running (max, sum, weighted-output) softmax statistics that
    get rescaled every time a new, larger max is seen (the standard online
    softmax trick).

    Causal masking is applied at TILE granularity:
      - key tiles strictly to the right of the query tile are fully
        upper-triangular (no allowed pairs) and are SKIPPED entirely,
      - the diagonal tile (same tile index for query and key) needs an
        elementwise lower-triangular mask,
      - key tiles strictly to the left of the query tile are fully allowed
        (no masking needed).

    Returns (n, d).
    """
    n, d = Q.shape
    num_blocks = (n + block_size - 1) // block_size

    O = np.zeros((n, d), dtype=np.float64)
    row_sum = np.zeros(n, dtype=np.float64)
    row_max = np.full(n, -float("inf"), dtype=np.float64)

    scale = 1.0 / math.sqrt(d)

    for it in range(num_blocks):
        i0, i1 = it * block_size, min((it + 1) * block_size, n)
        row_count_i = i1 - i0
        Qi = Q[i0:i1].astype(np.float64)
        Oi = O[i0:i1].copy()
        li = row_sum[i0:i1].copy()
        mi = row_max[i0:i1].copy()

        for jt in range(0, it + 1):
            j0, j1 = jt * block_size, min((jt + 1) * block_size, n)
            row_count_j = j1 - j0
            Kj = K[j0:j1].astype(np.float64)
            Vj = V[j0:j1].astype(np.float64)

            Sij = np.zeros((row_count_i, row_count_j), dtype=np.float64)
            for r in range(row_count_i):
                for c in range(row_count_j):
                    dot_val = 0.0
                    for k_idx in range(d):
                        dot_val += Qi[r, k_idx] * Kj[c, k_idx]
                    val = dot_val * scale
                    if jt == it and (j0 + c) > (i0 + r):
                        val = -float("inf")
                    Sij[r, c] = val

            m_block = np.full(row_count_i, -float("inf"), dtype=np.float64)
            for r in range(row_count_i):
                mx = -float("inf")
                for c in range(row_count_j):
                    if Sij[r, c] > mx:
                        mx = Sij[r, c]
                m_block[r] = mx

            m_new = np.zeros(row_count_i, dtype=np.float64)
            for r in range(row_count_i):
                if mi[r] > m_block[r]:
                    m_new[r] = mi[r]
                else:
                    m_new[r] = m_block[r]

            P = np.zeros((row_count_i, row_count_j), dtype=np.float64)
            for r in range(row_count_i):
                for c in range(row_count_j):
                    P[r, c] = math.exp(Sij[r, c] - m_new[r])

            alpha = np.zeros(row_count_i, dtype=np.float64)
            for r in range(row_count_i):
                alpha[r] = math.exp(mi[r] - m_new[r])

            p_sum = np.zeros(row_count_i, dtype=np.float64)
            for r in range(row_count_i):
                s_val = 0.0
                for c in range(row_count_j):
                    s_val += P[r, c]
                p_sum[r] = s_val

            new_li = np.zeros(row_count_i, dtype=np.float64)
            for r in range(row_count_i):
                new_li[r] = alpha[r] * li[r] + p_sum[r]
            li = new_li

            new_Oi = np.zeros((row_count_i, d), dtype=np.float64)
            for r in range(row_count_i):
                for col_idx in range(d):
                    pv_val = 0.0
                    for c in range(row_count_j):
                        pv_val += P[r, c] * Vj[c, col_idx]
                    new_Oi[r, col_idx] = alpha[r] * Oi[r, col_idx] + pv_val
            Oi = new_Oi
            mi = m_new

        final_Oi = np.zeros((row_count_i, d), dtype=np.float64)
        for r in range(row_count_i):
            inv_li = 1.0 / li[r]
            for col_idx in range(d):
                final_Oi[r, col_idx] = Oi[r, col_idx] * inv_li

        O[i0:i1] = final_Oi
        row_sum[i0:i1] = li
        row_max[i0:i1] = mi

    return O
