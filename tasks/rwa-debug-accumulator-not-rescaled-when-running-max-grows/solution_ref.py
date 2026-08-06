import math
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
    m = -float("inf")
    l = 0.0
    O = np.zeros(d, dtype=np.float64)

    sqrt_d = math.sqrt(d)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)

        scores = []
        for i in range(start, end):
            score = 0.0
            for j in range(d):
                score += q[j] * K[i, j]
            scores.append(score / sqrt_d)

        m_new = m
        for s in scores:
            if s > m_new:
                m_new = s

        correction = math.exp(m - m_new)

        p = []
        sum_p = 0.0
        for s in scores:
            val = math.exp(s - m_new)
            p.append(val)
            sum_p += val

        l = l * correction + sum_p

        block_len = end - start
        for j in range(d):
            v_acc = 0.0
            for k in range(block_len):
                v_acc += p[k] * V[start + k, j]
            O[j] = O[j] * correction + v_acc

        m = m_new

    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        out[j] = O[j] / l

    return out
