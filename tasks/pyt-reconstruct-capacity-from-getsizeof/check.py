import sys


def _oracle(n):
    empty_size = sys.getsizeof([])
    slots = []
    values = []
    for i in range(n):
        values.append(i)
        capacity = (sys.getsizeof(values) - empty_size) // 8
        slots.append(capacity)
    return slots


def grade(sol, fx) -> dict:
    cases = [0, 1, 2, 5, 16, 40, 100]
    ok = 1.0
    for n in cases:
        try:
            got = sol.reconstruct_capacity(n)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(n):
            ok = 0.0
            break
    return {"exact_match": ok}
