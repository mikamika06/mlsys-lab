from mlsys.sim import cache as cachesim

def grade(sol, fx) -> dict:
    S, d, B, elem_bytes = 32, 32, 8, 4
    try:
        addrs = list(sol.qkt_access_order(S, d, B, elem_bytes))
    except Exception:
        return {"modeled_cache_misses": 10**9}

    if len(addrs) == 0:
        return {"modeled_cache_misses": 10**9}

    result = cachesim.simulate(addrs, line_bytes=64, sets=64, ways=8)
    return {"modeled_cache_misses": result["misses"]}
