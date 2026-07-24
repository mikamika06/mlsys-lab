import math
import sys


def _oracle(pattern):
    requested = sum(pattern)
    if requested == 0:
        raise ValueError("empty allocation request")
    resident = 0
    objects = []
    try:
        for n in pattern:
            obj = bytearray(n)
            objects.append(obj)
            resident += sys.getsizeof(obj)
        return float(resident / requested)
    finally:
        objects.clear()


def grade(sol, fx) -> dict:
    cases = [
        [1, 2, 3, 4, 8, 16],
        [24, 25, 32, 33, 64, 65],
        [100, 128, 256, 512],
        [0, 1, 0, 7, 15, 31],
    ]
    ok = 1.0
    for pattern in cases:
        try:
            got = float(sol.internal_fragmentation_ratio(list(pattern)))
            ref = _oracle(list(pattern))
        except Exception:
            ok = 0.0
            break
        if not math.isclose(got, ref, rel_tol=0.0, abs_tol=0.0):
            ok = 0.0
            break
    return {"size_ratio": ok}
