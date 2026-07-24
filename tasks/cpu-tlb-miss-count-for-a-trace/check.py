from mlsys.sim import cache as cachesim


def _simulate_misses(addrs):
    result = cachesim.simulate(addrs, line_bytes=4096, sets=4, ways=4)
    if isinstance(result, dict):
        if "misses" in result:
            return result["misses"]
        if "miss_count" in result:
            return result["miss_count"]
    if hasattr(result, "misses"):
        return result.misses
    return result[0]


def _ref_trace(pages, rounds, page_size):
    return [i * page_size for _ in range(rounds) for i in range(pages)]


def grade(sol, fx) -> dict:
    cases = [
        (2, 8, 4096),
        (4, 4, 4096),
        (8, 3, 4096),
        (16, 2, 4096),
        (5, 7, 8192),
    ]

    ok = 1.0
    for pages, rounds, page_size in cases:
        try:
            got_trace = sol.make_tlb_trace(pages, rounds, page_size)
            got_trace = list(got_trace)
            got = _simulate_misses(got_trace)
            ref = _simulate_misses(_ref_trace(pages, rounds, page_size))
        except Exception:
            ok = 0.0
            break

        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
