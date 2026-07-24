import sys


def _oracle_count(n):
    sizes = []
    result = [None] * n
    sizes.append(sys.getsizeof(result))
    return sum(1 for a, b in zip(sizes, sizes[1:]) if a != b)


def grade(sol, fx) -> dict:
    cases = [0, 1, 8, 64, 1000, 4096]
    expected = [_oracle_count(n) for n in cases]
    ok = 1.0
    for n, ref in zip(cases, expected):
        try:
            got = sol.build_list_realloc_count(n)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
