import math
import numpy as np


def expected_attention_scores(queries, keys, top_k):
    queries = np.asarray(queries, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)

    N = queries.shape[0]
    d = queries.shape[1]
    M = keys.shape[0]

    mu = [0.0] * d
    for j in range(d):
        s = 0.0
        for i in range(N):
            s += queries[i, j]
        mu[j] = s / N

    centered = [[queries[i, j] - mu[j] for j in range(d)] for i in range(N)]

    cov = [[0.0] * d for _ in range(d)]
    denom_cov = N - 1
    for a in range(d):
        for b in range(d):
            s = 0.0
            for i in range(N):
                s += centered[i][a] * centered[i][b]
            cov[a][b] = s / denom_cov

    sqrt_d = math.sqrt(d)
    mean_term = [0.0] * M
    for i in range(M):
        s = 0.0
        for j in range(d):
            s += keys[i, j] * mu[j]
        mean_term[i] = s / sqrt_d

    variance_term = [0.0] * M
    for i in range(M):
        s = 0.0
        for j in range(d):
            for k in range(d):
                s += keys[i, j] * cov[j][k] * keys[i, k]
        variance_term[i] = s / d

    scores_list = [0.0] * M
    for i in range(M):
        scores_list[i] = mean_term[i] + 0.5 * variance_term[i]

    indexed = [(-scores_list[i], i) for i in range(M)]
    sorted_indexed = sorted(indexed, key=lambda x: x[0])
    order_list = [sorted_indexed[i][1] for i in range(top_k)]

    return np.array(scores_list, dtype=np.float64), np.array(order_list, dtype=np.int64)
