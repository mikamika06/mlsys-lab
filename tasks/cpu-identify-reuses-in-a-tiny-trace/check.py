import numpy as np
from mlsys.sim import cache as cachesim

def _reference_hits(trace: np.ndarray) -> int:
    result = cachesim.simulate(
        trace,
        line_bytes=8,
        sets=4,
        ways=2
    )
    return int(result['hits'])

def grade(sol, fx) -> dict:
    # Fixed trace used by the grader
    trace = np.array([0, 8, 16, 24, 32, 40, 48, 56,
                      0, 8, 16, 24], dtype=np.int64)
    try:
        got = sol.count_cache_hits(trace)
    except Exception:
        return {"exact_match": 0.0}
    ref = _reference_hits(trace)
    ok = 1.0 if got == ref else 0.0
    return {"exact_match": ok}
