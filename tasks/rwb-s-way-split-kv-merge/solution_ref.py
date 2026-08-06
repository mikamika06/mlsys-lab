import math
import numpy as np


def merge_split_kv(partials):
    ms_list = [np.asarray(p[0], dtype=np.float64) for p in partials]
    ls_list = [np.asarray(p[1], dtype=np.float64) for p in partials]
    accs_list = [np.asarray(p[2], dtype=np.float64) for p in partials]

    S = len(partials)
    n = ms_list[0].shape[0]
    d = accs_list[0].shape[1]

    m_global = [0.0] * n
    for j in range(n):
        max_val = ms_list[0][j]
        for s in range(1, S):
            val = ms_list[s][j]
            if val > max_val:
                max_val = val
        m_global[j] = max_val

    correction_list = []
    for s in range(S):
        corr_s = [0.0] * n
        for j in range(n):
            corr_s[j] = math.exp(ms_list[s][j] - m_global[j])
        correction_list.append(corr_s)

    l_global = [0.0] * n
    for j in range(n):
        s_val = 0.0
        for s in range(S):
            s_val += ls_list[s][j] * correction_list[s][j]
        l_global[j] = s_val

    acc_global = [[0.0] * d for _ in range(n)]
    for j in range(n):
        for k in range(d):
            s_val = 0.0
            for s in range(S):
                s_val += accs_list[s][j, k] * correction_list[s][j]
            acc_global[j][k] = s_val

    result = [[0.0] * d for _ in range(n)]
    for j in range(n):
        l_val = l_global[j]
        for k in range(d):
            result[j][k] = acc_global[j][k] / l_val

    return np.array(result, dtype=np.float64)
