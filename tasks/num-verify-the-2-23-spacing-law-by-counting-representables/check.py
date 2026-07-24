import numpy as np

def _ref_count(start, end):
    s = np.float32(start).view(np.uint32)
    e = np.float32(end).view(np.uint32)
    return int(e - s)

def grade(sol, fx) -> dict:
    try:
        got1 = sol.count_fp32_in_range(1.0, 2.0)
        got2 = sol.count_fp32_in_range(1024.0, 2048.0)
    except Exception:
        return {"exact_match": 0.0}
    ref1 = _ref_count(1.0, 2.0)
    ref2 = _ref_count(1024.0, 2048.0)
    ok = 1.0 if (got1 == ref1 and got2 == ref2) else 0.0
    return {"exact_match": ok}
