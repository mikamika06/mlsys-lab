from mlsys.sim import cache as cachesim


def _reference(n, tile):
    out = []
    for tr in range(0, n, tile):
        for tc in range(0, n, tile):
            for i in range(tr, min(tr + tile, n)):
                for j in range(tc, min(tc + tile, n)):
                    out.append((i * n + j) * 8)
    return out


def _misses(addrs):
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
        (4, 2),
        (5, 2),
        (8, 4),
        (9, 3),
        (16, 4),
    ]
    ok = 1.0
    for n, tile in cases:
        try:
            got = list(sol.blocked_access_trace(n, tile))
        except Exception:
            ok = 0.0
            break

        ref = _reference(n, tile)

        try:
            got_misses = _misses(got)
            ref_misses = _misses(ref)
        except Exception:
            ok = 0.0
            break

        if got != ref or got_misses != ref_misses:
            ok = 0.0
            break

    return {"exact_match": ok}
