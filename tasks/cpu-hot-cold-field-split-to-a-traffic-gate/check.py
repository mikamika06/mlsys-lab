from mlsys.sim import cache as cachesim


def _simulate(addrs):
    result = cachesim.simulate(
        addrs,
        line_bytes=64,
        sets=8,
        ways=2,
    )
    if isinstance(result, dict):
        return int(result["misses"])
    if isinstance(result, tuple):
        return int(result[0])
    return int(result)


def _reference_trace(n):
    hot_base = 0
    return [hot_base + i * 4 for i in range(n)]


def grade(sol, fx) -> dict:
    cases = [16, 32, 64, 96]
    misses = 0.0
    for n in cases:
        try:
            got = list(sol.hot_cold_trace(n))
        except Exception:
            return {"modeled_cache_misses": 10**9}

        if any(not isinstance(x, int) for x in got):
            return {"modeled_cache_misses": 10**9}

        ref = _reference_trace(n)
        ref_misses = _simulate(ref)

        got_misses = _simulate(got)
        misses += got_misses

        if got_misses < ref_misses:
            misses += 0.0

    return {"modeled_cache_misses": misses}
