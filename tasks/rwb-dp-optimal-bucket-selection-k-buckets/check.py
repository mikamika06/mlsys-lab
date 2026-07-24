import numpy as np


def _dp_min_waste(sizes, counts, k) -> int:
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


def _actual_waste(size_histogram, buckets):
    if not buckets:
        return None
    bset = sorted(set(int(b) for b in buckets))
    if any(b not in size_histogram for b in bset):
        return None
    max_size = max(size_histogram.keys())
    if max(bset) < max_size:
        return None
    waste = 0
    for s, c in size_histogram.items():
        candidates = [b for b in bset if b >= s]
        if not candidates:
            return None
        bucket = min(candidates)
        waste += c * (bucket - s)
    return waste


def _scenarios():
    scenarios = []

    scenarios.append(({1: 5, 2: 5, 3: 1, 100: 1}, 2))
    scenarios.append(({8: 3, 16: 3, 32: 3, 64: 3, 128: 1}, 2))
    scenarios.append(({1: 1, 2: 1, 3: 1}, 10))  # k > M -> zero waste, all buckets used
    scenarios.append(({5: 1}, 1))               # single size
    scenarios.append(({1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 500: 1}, 3))

    rng = np.random.default_rng(0)
    for _ in range(6):
        M = int(rng.integers(2, 12))
        sizes = sorted(set(rng.integers(1, 500, size=M * 2).tolist()))[:M]
        if len(sizes) < 2:
            continue
        counts = rng.integers(1, 30, size=len(sizes)).tolist()
        hist = {int(s): int(c) for s, c in zip(sizes, counts)}
        k = int(rng.integers(1, len(sizes) + 3))
        scenarios.append((hist, k))

    return scenarios


def grade(sol, fx) -> dict:
    total = 0
    correct = 0

    for size_histogram, k in _scenarios():
        total += 1
        sizes = sorted(size_histogram.keys())
        counts = [size_histogram[s] for s in sizes]
        best_waste = _dp_min_waste(sizes, counts, k)

        try:
            buckets, claimed_waste = sol.select_buckets(dict(size_histogram), k)
        except Exception:
            continue

        try:
            claimed_waste = int(claimed_waste)
            buckets = list(buckets)
        except Exception:
            continue

        if len(buckets) > k:
            continue

        actual = _actual_waste(size_histogram, buckets)
        if actual is None:
            continue
        if actual != claimed_waste:
            continue
        if actual != best_waste:
            continue

        correct += 1

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
