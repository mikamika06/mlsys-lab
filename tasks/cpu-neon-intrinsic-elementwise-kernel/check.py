from mlsys.sim import cache as cachesim


def _reference_kernel(n):
    out = bytearray()
    for i in range(n):
        out.extend(int(3 * i + 7).to_bytes(4, "little", signed=True))

    trace = []
    for i in range(n):
        trace.append(i * 4)
    for i in range(n):
        trace.append(4096 + i * 4)
    return bytes(out), trace


def _misses(result):
    if isinstance(result, dict):
        for key in ("misses", "miss_count", "cache_misses"):
            if key in result:
                return int(result[key])
    if hasattr(result, "misses"):
        return int(result.misses)
    return int(result)


def grade(sol, fx) -> dict:
    cases = [1, 7, 32, 128]
    exact = 1.0
    max_misses = 0
    for n in cases:
        try:
            got_bytes, got_trace = sol.elementwise_kernel(n)
        except Exception:
            return {
                "byte_exact_fraction": 0.0,
                "cache_miss_count": 10**9,
            }

        ref_bytes, ref_trace = _reference_kernel(n)

        if bytes(got_bytes) != ref_bytes:
            exact = 0.0

        misses = _misses(
            cachesim.simulate(
                list(got_trace),
                line_bytes=64,
                sets=4,
                ways=2,
            )
        )
        max_misses = max(max_misses, misses)

    return {
        "byte_exact_fraction": exact,
        "cache_miss_count": max_misses,
    }
