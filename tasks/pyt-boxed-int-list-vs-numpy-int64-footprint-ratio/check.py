import sys

import numpy as np


def _ref(values: list) -> float:
    list_bytes = sys.getsizeof(values)
    seen = {}
    for v in values:
        seen[id(v)] = v
    list_bytes += sum(sys.getsizeof(v) for v in seen.values())

    arr = np.array(values, dtype=np.int64)
    array_bytes = sys.getsizeof(arr)
    return list_bytes / array_bytes if array_bytes else 0.0


def _make_case(rng, n_small_repeat=20, n_unique_large=500):
    values = list(range(-5, 257))              # every cached small int, once each
    values.extend([100] * n_small_repeat)       # 20 more refs to the SAME cached object
    start = int(rng.integers(10_000, 1_000_000))
    values.extend(range(start, start + n_unique_large))
    rng.shuffle(values)
    return values


def _cases():
    rng = np.random.default_rng(0)
    cases = [_make_case(rng) for _ in range(3)]
    cases.append([])
    cases.append([0])
    cases.append([2 ** 62, -(2 ** 61), 2 ** 40, -5, 256, 257])
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for values in _cases():
        ref = _ref(list(values))
        try:
            got = float(sol.list_footprint_ratio(list(values)))
        except Exception:
            return {"rel_err": float("inf")}
        if not np.isfinite(got):
            return {"rel_err": float("inf")}
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)
    return {"rel_err": worst}
