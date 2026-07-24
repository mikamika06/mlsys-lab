from mlsys.sim import cache as cachesim


def _ref_distance(mem_latency, loop_body_cycles):
    return (mem_latency + loop_body_cycles - 1) // loop_body_cycles


def _trace_for_distance(distance):
    addrs = []
    stride = 64
    total = 128
    for i in range(total):
        addrs.append((i * stride) & ((distance + 8) * 4096 - 1))
        if i >= distance:
            addrs.append(((i - distance) * stride) & ((distance + 8) * 4096 - 1))
    return addrs


def _cache_score(distance):
    addrs = _trace_for_distance(distance)
    result = cachesim.simulate(addrs, line_bytes=64, sets=32, ways=4)
    if isinstance(result, dict):
        return result.get("misses", result.get("miss_count", result))
    return result


def grade(sol, fx) -> dict:
    cases = [
        (64, 8),
        (127, 16),
        (128, 16),
        (511, 32),
        (1000, 37),
        (4096, 128),
    ]
    ok = 1.0
    for latency, body in cases:
        try:
            got = sol.optimal_prefetch_distance(latency, body)
            ref = _ref_distance(latency, body)
            _cache_score(ref)
            _cache_score(got)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
