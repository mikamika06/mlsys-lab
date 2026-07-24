import numpy as np


def _dp_min_waste(sizes, counts, k) -> int:
    M = len(sizes)
    k_eff = min(k, M)
    cum_c = [0] * (M + 1)
    cum_cs = [0] * (M + 1)
    for i in range(M):
        cum_c[i + 1] = cum_c[i] + counts[i]
        cum_cs[i + 1] = cum_cs[i] + counts[i] * sizes[i]

    def cost(i, j):
        sum_c = cum_c[j + 1] - cum_c[i]
        sum_cs = cum_cs[j + 1] - cum_cs[i]
        return sizes[j] * sum_c - sum_cs

    INF = float("inf")
    dp = [[INF] * M for _ in range(k_eff + 1)]
    for j in range(M):
        dp[1][j] = cost(0, j)
    for kk in range(2, k_eff + 1):
        for j in range(kk - 1, M):
            best = INF
            for i in range(kk - 2, j):
                v = dp[kk - 1][i] + cost(i + 1, j)
                if v < best:
                    best = v
            dp[kk][j] = best
    return int(dp[k_eff][M - 1])


def compare_k4_k8_waste(sizes: np.ndarray, counts: np.ndarray):
    """
    Compute the minimum achievable padding waste with K=4 buckets and with
    K=8 buckets over the same observed size histogram (via the optimal
    contiguous-partition DP), and the waste reduction from using more
    buckets.

    Returns (waste_k4, waste_k8, reduction) with reduction = waste_k4 - waste_k8.
    """
    sizes = [int(s) for s in np.asarray(sizes).tolist()]
    counts = [int(c) for c in np.asarray(counts).tolist()]
    pairs = sorted(zip(sizes, counts))
    sorted_sizes = [p[0] for p in pairs]
    sorted_counts = [p[1] for p in pairs]

    waste_k4 = _dp_min_waste(sorted_sizes, sorted_counts, 4)
    waste_k8 = _dp_min_waste(sorted_sizes, sorted_counts, 8)
    return waste_k4, waste_k8, waste_k4 - waste_k8
