def select_buckets(size_histogram, k):
    sizes = sorted(size_histogram.keys())
    counts = [size_histogram[s] for s in sizes]
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
    choice = [[-1] * M for _ in range(k_eff + 1)]
    for j in range(M):
        dp[1][j] = cost(0, j)
        choice[1][j] = -1

    for kk in range(2, k_eff + 1):
        for j in range(kk - 1, M):
            best = INF
            best_i = -1
            for i in range(kk - 2, j):
                v = dp[kk - 1][i] + cost(i + 1, j)
                if v < best:
                    best = v
                    best_i = i
            dp[kk][j] = best
            choice[kk][j] = best_i

    total_waste = int(dp[k_eff][M - 1])

    # backtrack to recover the chosen bucket boundaries
    buckets = []
    j = M - 1
    kk = k_eff
    while kk >= 1:
        buckets.append(sizes[j])
        i = choice[kk][j]
        j = i
        kk -= 1
    buckets.sort()

    return buckets, total_waste
