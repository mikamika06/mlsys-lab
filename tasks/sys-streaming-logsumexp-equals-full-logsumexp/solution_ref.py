import math


def streaming_logsumexp(chunks: list[list[list[float]]]) -> list[float]:
    N = len(chunks[0])

    m = [-float("inf")] * N
    ell = [0.0] * N

    for c in chunks:
        nrows = len(c)
        ncols = len(c[0])
        for i in range(nrows):
            c_max = -float("inf")
            for j in range(ncols):
                val = c[i][j]
                if val > c_max:
                    c_max = val
            m_old = m[i]
            m_new = m_old if m_old > c_max else c_max
            term = 0.0
            for j in range(ncols):
                term += math.exp(c[i][j] - m_new)
            ell[i] = ell[i] * math.exp(m_old - m_new) + term
            m[i] = m_new

    res = [0.0] * N
    for i in range(N):
        res[i] = m[i] + math.log(ell[i])
    return res
