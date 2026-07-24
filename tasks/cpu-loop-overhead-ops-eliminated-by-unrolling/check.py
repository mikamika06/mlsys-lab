from mlsys.sim import cache as cachesim

def grade(sol, fx) -> dict:
    N, U, elem_bytes = 1024, 4, 8
    try:
        saved, trace = sol.unroll_overhead(N, U)
    except Exception:
        return {"exact_match": 0.0, "covers_all": 0.0, "misses": 10**9}

    # ---- exact_match: compare against the algebraic formula ----
    expected_saved = 2 * (N - N // U)
    exact = 1.0 if saved == expected_saved else 0.0

    # ---- covers_all: trace must be a permutation of all N byte addresses ----
    expected_addrs = [i * elem_bytes for i in range(N)]
    covers = 1.0 if sorted(trace) == expected_addrs else 0.0

    # ---- misses: deterministic cache simulation on the returned trace ----
    try:
        result = cachesim.simulate(trace, line_bytes=64, sets=64, ways=8)
        misses = result["misses"]
    except Exception:
        misses = 10**9

    return {"exact_match": exact, "covers_all": covers, "misses": misses}
