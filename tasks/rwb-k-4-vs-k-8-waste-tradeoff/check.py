import numpy as np


def _dp_min_waste(sizes, counts, k) -> int:
    """Real oracle: exact DP over contiguous partitions of the sorted
    distinct sizes into `k` bucket ranges, each covered by its range's
    largest size (see rwb-dp-optimal-bucket-selection-k-buckets for the
    derivation of why contiguous ranges are optimal here).
    """
    M = len(sizes)
    k_eff = min(k, M)
    cum_c = [0] * (M + 1)
    cum_cs = [0] * (M + 1)
    for i in range(M):
        cum_c[i + 1] = cum_c[i] + counts[i]
        cum_cs[i + 1] = cum_cs[i] + counts[i] * sizes[i]

    def cost(i, j):  # inclusive 0-indexed range covered by bucket sizes[j]
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


def _oracle(sizes, counts):
    sizes = [int(s) for s in sizes]
    counts = [int(c) for c in counts]
    pairs = sorted(zip(sizes, counts))
    sorted_sizes = [p[0] for p in pairs]
    sorted_counts = [p[1] for p in pairs]

    waste_k4 = _dp_min_waste(sorted_sizes, sorted_counts, 4)
    waste_k8 = _dp_min_waste(sorted_sizes, sorted_counts, 8)
    reduction = waste_k4 - waste_k8
    assert reduction >= 0, "more buckets can never increase optimal waste"
    return waste_k4, waste_k8, reduction


def _extra_cases():
    rng = np.random.default_rng(2)
    cases = []
    for _ in range(4):
        M = int(rng.integers(9, 25))  # comfortably more than 8 distinct sizes
        sizes = sorted(set(rng.integers(1, 1000, size=M * 2).tolist()))[:M]
        counts = rng.integers(1, 60, size=len(sizes)).tolist()
        cases.append((sizes, counts))
    # exactly 8 distinct sizes: K=8 should hit zero waste, K=4 should not (usually)
    sizes8 = sorted(rng.choice(np.arange(1, 500), size=8, replace=False).tolist())
    counts8 = rng.integers(1, 40, size=8).tolist()
    cases.append((sizes8, counts8))
    return cases


def grade(sol, fx) -> dict:
    cases = [(np.asarray(fx["sizes"]), np.asarray(fx["counts"]))] + _extra_cases()

    total = 0
    correct = 0
    for sizes, counts in cases:
        total += 1
        ref = _oracle(sizes, counts)
        try:
            got = sol.compare_k4_k8_waste(np.asarray(sizes).copy(), np.asarray(counts).copy())
            got_w4, got_w8, got_reduction = (int(v) for v in got)
        except Exception:
            continue

        if (got_w4, got_w8, got_reduction) == ref:
            correct += 1

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
