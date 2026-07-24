import sys
import numpy as np

def _reference_order():
    reg = 0
    sh = [0] * 10
    l2 = np.zeros(10)
    glob = {i: 0 for i in range(10)}
    sizes = {
        'register': sys.getsizeof(reg),
        'shared': sys.getsizeof(sh),
        'L2': sys.getsizeof(l2),
        'global': sys.getsizeof(glob)
    }
    return sorted(sizes, key=sizes.get)

def grade(sol, fx) -> dict:
    try:
        got = sol.rank_memory_spaces()
    except Exception:
        return {"exact_match": 0.0}
    ref = _reference_order()
    ok = 1.0 if got == ref else 0.0
    return {"exact_match": ok}
