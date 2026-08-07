import math


def tiled_online_softmax_attention(q: list[float], K: list[list[float]], V: list[list[float]], block_size: int) -> list[float]:
    """FlashAttention-style single-query forward pass: stream over K/V in
    blocks of `block_size`, maintaining a running max `m`, running
    normalizer `l`, and an UNNORMALIZED output accumulator `O`. Every time
    a new block raises the running max, both `l` AND `O` must be rescaled
    by exp(m_old - m_new) before adding the new block's contribution --
    otherwise earlier blocks stay weighted against their own stale local
    max instead of the final global max. Returns O / l, shape (d,).
    """
    d = len(q)
    n = len(K)
    m = -float("inf")
    l = 0.0
    O = [0.0] * d

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        Kb = K[start:end]
        Vb = V[start:end]

        scores = [sum(qi * kij for qi, kij in zip(q, ki)) / math.sqrt(d) for ki in Kb]
        m_new = max(m, max(scores))
        correction = math.exp(m - m_new)  # 0.0 on the first block (m == -inf)

        p = [math.exp(s - m_new) for s in scores]
        sum_p = sum(p)
        l = l * correction + sum_p

        O = [o_val * correction + sum(pi * vb[j] for pi, vb in zip(p, Vb)) for j, o_val in enumerate(O)]

        m = m_new

    return [o_val / l for o_val in O]
