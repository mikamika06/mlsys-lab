import numpy as np
from mlsys.sim import cache as cachesim

def _build_trace(n: int, store_type: str) -> list[int]:
    line_bytes = 64
    addr_per_float = 4
    addrs = []
    seen_lines = set()
    for i in range(n):
        addr = i * addr_per_float
        line_base = (addr // line_bytes) * line_bytes
        if store_type == "temporal":
            # simulate read-for-ownership for the line on first touch
            if line_base not in seen_lines:
                seen_lines.add(line_base)
                for j in range(0, line_bytes, addr_per_float):
                    addrs.append(line_base + j)  # read phase
        # actual write
        addrs.append(addr)
    return addrs


def _modeled_misses(addrs):
    # deterministic nominal cache parameters
    return cachesim.simulate(addrs, line_bytes=64, sets=64, ways=8)["misses"]


def grade(sol, fx) -> dict:
    tests = [256, 1024, 4096]
    miss_ratios = []
    ok = 1.0
    for n in tests:
        ref_trace = _build_trace(n, "non-temporal")
        got_trace = sol.simulate_activation_write(n, "non-temporal")
        # safety: must be list[int]
        if not isinstance(got_trace, (list, tuple)):
            return {"modeled_cache_misses": 9999.0}
        ref_misses = _modeled_misses(ref_trace)
        try:
            got_misses = _modeled_misses(got_trace)
        except Exception:
            ok = 0.0
            got_misses = 1e12
        miss_ratios.append(got_misses / (ref_misses + 1e-12))
    avg_ratio = float(np.mean(miss_ratios))
    return {"modeled_cache_misses": avg_ratio}
