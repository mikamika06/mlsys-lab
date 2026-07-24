import random as _random
from mlsys.sim import cache as cachesim

def grade(sol, fx) -> dict:
    m, vw, e = 128, 4, 8
    line_bytes, sets, ways = 64, 64, 8

    # ---- reference answers computed with the algorithm (never hardcoded) ----
    ref_contiguous_uops = (m + vw - 1) // vw
    ref_gather_uops = m

    cont_addrs = [i * e for i in range(m)]
    ref_contiguous_misses = cachesim.simulate(
        cont_addrs, line_bytes=line_bytes, sets=sets, ways=ways
    )["misses"]

    indices = list(range(m))
    _random.seed(42)
    _random.shuffle(indices)
    gath_addrs = [i * e for i in indices]
    ref_gather_misses = cachesim.simulate(
        gath_addrs, line_bytes=line_bytes, sets=sets, ways=ways
    )["misses"]

    # ---- run learner ----
    try:
        r = sol.modeled_load_uops(m, vw, e)
        if not isinstance(r, dict):
            raise TypeError("must return dict")
        learner = {
            "contiguous_uops":  int(r["contiguous_uops"]),
            "gather_uops":      int(r["gather_uops"]),
            "contiguous_misses": int(r["contiguous_misses"]),
            "gather_misses":     int(r["gather_misses"]),
        }
    except Exception:
        return {k: 0 for k in ["contiguous_uops", "gather_uops",
                                 "contiguous_misses", "gather_misses"]}

    return {
        "contiguous_uops":  1 if learner["contiguous_uops"]  == ref_contiguous_uops  else 0,
        "gather_uops":      1 if learner["gather_uops"]      == ref_gather_uops      else 0,
        "contiguous_misses":1 if learner["contiguous_misses"] == ref_contiguous_misses else 0,
        "gather_misses":    1 if learner["gather_misses"]     == ref_gather_misses     else 0,
    }
