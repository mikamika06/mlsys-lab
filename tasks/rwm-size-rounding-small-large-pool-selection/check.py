import math
import numpy as np

MIN_BLOCK_SIZE = 512
SMALL_SIZE = 1024 * 1024
SMALL_BUFFER = 2 * 1024 * 1024
LARGE_BUFFER = 20 * 1024 * 1024
MIN_LARGE_ALLOC = 10 * 1024 * 1024
ROUND_LARGE = 2 * 1024 * 1024


def _round_request(nbytes):
    n = max(int(nbytes), 1)
    return MIN_BLOCK_SIZE * math.ceil(n / MIN_BLOCK_SIZE)


def _segment_size(rounded):
    if rounded <= SMALL_SIZE:
        return SMALL_BUFFER
    if rounded < MIN_LARGE_ALLOC:
        return LARGE_BUFFER
    return ROUND_LARGE * math.ceil(rounded / ROUND_LARGE)


def _oracle(nbytes):
    rounded = _round_request(nbytes)
    pool = "small" if rounded <= SMALL_SIZE else "large"
    return pool, _segment_size(rounded)


def _cases():
    # Fixed boundary cases around every threshold in the algorithm.
    cases = [
        1, 100, 511, 512, 513, 1000, 1024,
        SMALL_SIZE - 1, SMALL_SIZE, SMALL_SIZE + 1,
        SMALL_SIZE + MIN_BLOCK_SIZE,
        MIN_LARGE_ALLOC - 1, MIN_LARGE_ALLOC, MIN_LARGE_ALLOC + 1,
        MIN_LARGE_ALLOC + 12345,
        2 * MIN_LARGE_ALLOC, 2 * MIN_LARGE_ALLOC + 1,
        ROUND_LARGE * 7, ROUND_LARGE * 7 + 1,
    ]

    rng = np.random.default_rng(0)
    for lo, hi, n in [
        (1, 2000, 40),
        (2000, SMALL_SIZE, 40),
        (SMALL_SIZE, MIN_LARGE_ALLOC, 40),
        (MIN_LARGE_ALLOC, 200 * 1024 * 1024, 40),
    ]:
        cases.extend(int(x) for x in rng.integers(lo, hi, size=n))

    return cases


def grade(sol, fx) -> dict:
    for nbytes in _cases():
        ref = _oracle(nbytes)
        try:
            got = sol.route_allocation(nbytes)
            got_pool, got_seg = got
        except Exception:
            return {"exact_match": 0.0}

        if (got_pool, int(got_seg)) != ref:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
