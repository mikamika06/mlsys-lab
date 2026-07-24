import sys
from typing import Iterable, List

def _reference_ratio(elements: Iterable[int]) -> float:
    s = set(elements)
    d = {e: e for e in elements}
    return sys.getsizeof(s) / sys.getsizeof(d)

def grade(sol, fx) -> dict:
    cases: List[List[int]] = [
        list(range(5)),          # small distinct
        [1, 2, 3, 4, 5],         # same as above but explicit
        [0, 1, 2, 2, 3, 3, 4],   # duplicates
        [10**6 + i for i in range(100)],  # larger numbers
    ]
    ok = 1.0
    for elements in cases:
        try:
            got = sol.set_dict_size_ratio(elements)
            ref = _reference_ratio(elements)
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, (float, int)):
            ok = 0.0
            break
        rel_err = abs(got - ref) / max(abs(ref), 1e-12)
        if rel_err > 1e-9:
            ok = 0.0
            break
    return {"size_ratio": ok}
