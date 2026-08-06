def online_softmax_stats(S):
    import numpy as np
    import math

    n, d = S.shape
    m_list = []
    l_list = []

    for i in range(n):
        m_i = S[i, 0]
        for j in range(1, d):
            val = S[i, j]
            if val > m_i:
                m_i = val
        m_list.append(m_i)

        l_i = 0.0
        for j in range(d):
            l_i += math.exp(S[i, j] - m_i)
        l_list.append(l_i)

    return np.array(m_list, dtype=np.float64), np.array(l_list, dtype=np.float64)
