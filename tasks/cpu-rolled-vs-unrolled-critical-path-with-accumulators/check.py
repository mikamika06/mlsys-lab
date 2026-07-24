from mlsys.sim import cache as cachesim


def _reference(n, k):
    latency = 4
    rolled = n * latency
    unrolled = ((n + k - 1) // k) * latency
    addresses = [index * 8 for index in range(n)]
    return rolled, unrolled, addresses


def _cache_misses(addrs):
    result = cachesim.simulate(
        addrs,
        line_bytes=64,
        sets=16,
        ways=2,
    )
    if isinstance(result, dict):
        return int(result["misses"])
    return int(result.misses)


def grade(sol, fx) -> dict:
    cases = [
        (8, 4),
        (31, 4),
        (64, 8),
        (100, 3),
        (257, 16),
    ]

    exact_match = 1.0
    miss_count = 0.0

    for n, k in cases:
        ref = _reference(n, k)
        try:
            rolled, unrolled, addresses = sol.model_kernel(n, k)
        except Exception:
            exact_match = 0.0
            miss_count = 1.0
            break

        if (rolled, unrolled) != (ref[0], ref[1]):
            exact_match = 0.0

        try:
            candidate_misses = _cache_misses(list(addresses))
            reference_misses = _cache_misses(ref[2])
        except Exception:
            miss_count = 1.0
            continue

        miss_count += float(candidate_misses != reference_misses)

    if miss_count != 0.0:
        miss_count = 1.0

    return {
        "exact_match": exact_match,
        "miss_count": miss_count,
    }
