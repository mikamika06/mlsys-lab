from mlsys.sim import cache as cachesim


def _balanced_depth(n):
    depth = 0
    width = n
    while width > 1:
        width = (width + 1) // 2
        depth += 1
    return depth


def _ref(values, base_addr):
    addrs = [base_addr + i * 8 for i in range(len(values))]
    total = float(sum(values))
    critical_path = _balanced_depth(len(values))
    return {
        "total": total,
        "addrs": addrs,
        "critical_path": critical_path,
    }


def _cache_misses(addrs):
    result = cachesim.simulate(
        addrs,
        line_bytes=64,
        sets=8,
        ways=2,
    )
    if isinstance(result, dict):
        return result["misses"]
    return result.misses


def grade(sol, fx) -> dict:
    cases = [
        ([1.0, 2.0, 3.0, 4.0], 4096),
        ([0.5, -1.5, 2.0, 8.0, 3.0], 8192),
        (list(range(17)), 16384),
        ([7.0], 32768),
    ]

    ok = 1.0
    for values, base_addr in cases:
        try:
            got = sol.reassociated_sum_trace(list(values), base_addr)
        except Exception:
            ok = 0.0
            break

        ref = _ref(values, base_addr)

        if got != ref:
            ok = 0.0
            break

        if _cache_misses(got["addrs"]) != _cache_misses(ref["addrs"]):
            ok = 0.0
            break

    return {"exact_match": ok}
