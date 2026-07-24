from math import ceil
from mlsys.sim import cache as cachesim


def _simulate_misses(addrs, line_bytes, sets, ways):
    result = cachesim.simulate(addrs, line_bytes=line_bytes, sets=sets, ways=ways)
    if isinstance(result, dict):
        return int(result["misses"])
    return int(result.misses)


def _reference_trace(length, line_bytes):
    return [i * line_bytes for i in range(length)]


def grade(sol, fx) -> dict:
    cases = [
        (5, 0.5, 4, 64, 8, 2),
        (7, 1.0, 16, 64, 4, 2),
        (3, 2.0, 32, 128, 8, 4),
        (11, 0.25, 9, 32, 16, 1),
    ]

    exact_match = 1.0
    cache_misses = 1.0

    for latency, throughput, length, line_bytes, sets, ways in cases:
        try:
            got_acc, got_addrs = sol.min_fma_accumulators(
                latency, throughput, length, line_bytes, sets, ways
            )
            got_addrs = list(got_addrs)
        except Exception:
            exact_match = 0.0
            cache_misses = 0.0
            break

        ref_acc = ceil(latency * throughput)
        ref_addrs = _reference_trace(length, line_bytes)

        if got_acc != ref_acc:
            exact_match = 0.0

        got_misses = _simulate_misses(got_addrs, line_bytes, sets, ways)
        ref_misses = _simulate_misses(ref_addrs, line_bytes, sets, ways)

        if got_misses != ref_misses or got_addrs != ref_addrs:
            cache_misses = 0.0

    return {
        "exact_match": exact_match,
        "cache_misses": cache_misses,
    }
