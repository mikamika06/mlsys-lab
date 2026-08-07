import math


def merge_split_kv(partials):
    S = len(partials)
    n = len(partials[0][0])
    d = len(partials[0][2][0])

    m_global = [0.0] * n
    for j in range(n):
        max_val = partials[0][0][j]
        for s in range(1, S):
            val = partials[s][0][j]
            if val > max_val:
                max_val = val
        m_global[j] = max_val

    correction_list = []
    for s in range(S):
        corr_s = [0.0] * n
        for j in range(n):
            corr_s[j] = math.exp(partials[s][0][j] - m_global[j])
        correction_list.append(corr_s)

    l_global = [0.0] * n
    for j in range(n):
        s_val = 0.0
        for s in range(S):
            s_val += partials[s][1][j] * correction_list[s][j]
        l_global[j] = s_val

    acc_global = [[0.0] * d for _ in range(n)]
    for j in range(n):
        for k in range(d):
            s_val = 0.0
            for s in range(S):
                s_val += partials[s][2][j][k] * correction_list[s][j]
            acc_global[j][k] = s_val

    result = [[0.0] * d for _ in range(n)]
    for j in range(n):
        l_val = l_global[j]
        for k in range(d):
            result[j][k] = acc_global[j][k] / l_val

    return result
