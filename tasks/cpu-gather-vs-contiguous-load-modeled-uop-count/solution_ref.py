import random
from mlsys.sim import cache as cachesim

def modeled_load_uops(m, vw, e):
    """Return dict with modeled uop counts and simulated cache miss counts."""
    line_bytes, sets, ways = 64, 64, 8

    contiguous_uops = (m + vw - 1) // vw
    gather_uops = m

    cont_addrs = [i * e for i in range(m)]
    contiguous_misses = cachesim.simulate(
        cont_addrs, line_bytes=line_bytes, sets=sets, ways=ways
    )["misses"]

    indices = list(range(m))
    random.seed(42)
    random.shuffle(indices)
    gath_addrs = [i * e for i in indices]
    gather_misses = cachesim.simulate(
        gath_addrs, line_bytes=line_bytes, sets=sets, ways=ways
    )["misses"]

    return {
        "contiguous_uops": contiguous_uops,
        "gather_uops": gather_uops,
        "contiguous_misses": contiguous_misses,
        "gather_misses": gather_misses,
    }
