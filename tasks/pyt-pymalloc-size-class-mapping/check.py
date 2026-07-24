import numpy as np

ALIGN = 8
THRESHOLD = 512


def _ref_index(n):
    if 1 <= n <= THRESHOLD:
        return (n - 1) // ALIGN
    return -1


def grade(sol, fx) -> dict:
    boundary = []
    for k in range(1, THRESHOLD // ALIGN + 3):
        base = k * ALIGN
        boundary.extend([base - 1, base, base + 1])
    boundary = [b for b in boundary if b >= 1]

    rng = np.random.default_rng(0)
    random_sizes = [int(x) for x in rng.integers(1, 4000, size=40)]

    sizes = boundary + random_sizes
    expected = [_ref_index(n) for n in sizes]

    try:
        got = sol.pymalloc_size_class(list(sizes))
    except Exception:
        return {"exact_match": 0.0}

    try:
        got = [int(x) for x in got]
    except Exception:
        return {"exact_match": 0.0}

    ok = 1.0 if got == expected else 0.0
    return {"exact_match": ok}
