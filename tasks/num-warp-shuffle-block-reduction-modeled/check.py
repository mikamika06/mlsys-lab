import math
import numpy as np


def _model_access_count(block_size, warp_size):
    warps = math.ceil(block_size / warp_size)
    stages = int(math.log2(warp_size))
    count = block_size
    count += warps
    count += warps * warp_size * stages
    if warps > 1:
        count += warps
        count += warp_size * stages
    return count


def grade(sol, fx) -> dict:
    cases = [
        ([1.0, 2.0, 3.0, 4.0], 4, 4),
        ([0.5, -1.5, 2.0, 8.0, 3.0, 1.0, 4.0, 5.0], 8, 4),
        ([float(i) for i in range(16)], 16, 4),
        ([1.0] * 32, 32, 8),
    ]

    exact = 1.0
    access_ok = 1.0

    for values, block_size, warp_size in cases:
        expected = float(np.sum(np.asarray(values, dtype=np.float64)))
        expected_access = _model_access_count(block_size, warp_size)
        try:
            got, accesses = sol.block_reduce(list(values), block_size, warp_size)
        except Exception:
            exact = 0.0
            access_ok = 0.0
            break

        if not np.isclose(float(got), expected, rtol=1e-12, atol=1e-12):
            exact = 0.0
        if int(accesses) != expected_access:
            access_ok = 0.0

    return {
        "exact_match": exact,
        "modeled_access_count": access_ok,
    }
