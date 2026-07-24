import numpy as np


def tiled_online_softmax_attention(q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """FlashAttention-style single-query forward pass: stream over K/V in
    blocks of `block_size`, maintaining a running max `m`, running
    normalizer `l`, and an UNNORMALIZED output accumulator `O`. Every time
    a new block raises the running max, both `l` AND `O` must be rescaled
    by exp(m_old - m_new) before adding the new block's contribution --
    otherwise earlier blocks stay weighted against their own stale local
    max instead of the final global max. Returns O / l, shape (d,).
    """
    d = K.shape[1]
    n = K.shape[0]
    m = -np.inf
    l = 0.0
    O = np.zeros(d, dtype=np.float64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        Kb = K[start:end]
        Vb = V[start:end]

        scores = (q @ Kb.T) / np.sqrt(d)
        m_new = max(m, float(np.max(scores)))
        correction = np.exp(m - m_new)  # 0.0 on the first block (m == -inf)

        p = np.exp(scores - m_new)
        l = l * correction + float(np.sum(p))
        O = O * correction + p @ Vb  # <-- the accumulator rescale

        m = m_new

    return O / l
