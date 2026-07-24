import sys


def _oracle(n, cycles):
    d = {i: i for i in range(n)}
    before = sys.getsizeof(d)
    next_key = n
    for _ in range(cycles):
        for k in list(d):
            del d[k]
        for _ in range(n):
            d[next_key] = next_key
            next_key += 1
    after = sys.getsizeof(d)
    return before, after


def grade(sol, fx) -> dict:
    cases = [
        (8, 4),
        (100, 3),
        (1000, 5),
        (4096, 2),
    ]

    ok = 1.0
    for n, cycles in cases:
        try:
            got = sol.dict_footprint_churn(n, cycles)
        except Exception:
            ok = 0.0
            break

        if not isinstance(got, tuple) or len(got) != 2:
            ok = 0.0
            break

        ref_before, ref_after = _oracle(n, cycles)
        try:
            got_ratio = got[1] / got[0]
            ref_ratio = ref_after / ref_before
        except Exception:
            ok = 0.0
            break

        if abs(got_ratio - ref_ratio) > 1e-12:
            ok = 0.0
            break

        if got != (ref_before, ref_after):
            ok = 0.0
            break

    return {"size_ratio": ok}
