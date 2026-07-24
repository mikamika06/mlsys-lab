import sys


def _oracle(values):
    s = set()
    out = []
    for value in values:
        s.add(value)
        out.append(sys.getsizeof(s))
    return out


def grade(sol, fx) -> dict:
    cases = [
        list(range(20)),
        [5, 5, 5, 5, 5],
        list(range(100)),
        [17, 9, 17, 9, 25, 33, 41, 49],
    ]

    ok = 1.0
    for values in cases:
        try:
            got = sol.set_allocation_trace(list(values))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(values):
            ok = 0.0
            break

    return {"exact_match": ok}
