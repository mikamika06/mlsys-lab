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
    row_max = np.full(n, -np.inf, dtype=np.float64)

    for it in range(num_blocks):
        i0, i1 = it * block_size, min((it + 1) * block_size, n)
        Qi = Q[i0:i1].astype(np.float64)
        Oi = O[i0:i1].copy()
        li = row_sum[i0:i1].copy()
        mi = row_max[i0:i1].copy()

        for jt in range(0, it + 1):  # skip jt > it: fully masked, no work needed
            j0, j1 = jt * block_size, min((jt + 1) * block_size, n)
            Kj = K[j0:j1].astype(np.float64)
            Vj = V[j0:j1].astype(np.float64)

            Sij = (Qi @ Kj.T) / np.sqrt(d)

            if jt == it:  # diagonal tile: elementwise lower-triangular mask
                local_row = np.arange(i1 - i0)[:, None]
                local_col = np.arange(j1 - j0)[None, :]
                Sij = np.where(local_col <= local_row, Sij, -np.inf)

            m_block = np.max(Sij, axis=1)
            m_new = np.maximum(mi, m_block)
            P = np.exp(Sij - m_new[:, None])
            alpha = np.exp(mi - m_new)
            li = alpha * li + np.sum(P, axis=1)
            Oi = alpha[:, None] * Oi + P @ Vj
            mi = m_new

        O[i0:i1] = Oi / li[:, None]
        row_sum[i0:i1] = li
        row_max[i0:i1] = mi

    return O
