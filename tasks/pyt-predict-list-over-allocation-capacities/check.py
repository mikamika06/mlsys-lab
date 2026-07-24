import sys


def _capacity(lst):
    empty_size = sys.getsizeof([])
    ptr_size = sys.getsizeof([None]) - empty_size
    return (sys.getsizeof(lst) - empty_size) // ptr_size


def _oracle(n):
    out = []
    lst = []
    last = _capacity(lst)
    for i in range(n):
        lst.append(i)
        cap = _capacity(lst)
        if cap != last:
            out.append(cap)
            last = cap
    return out


def grade(sol, fx) -> dict:
    cases = [0, 1, 2, 4, 5, 8, 9, 16, 17, 32, 64, 100, 256]
    ok = 1.0
    for n in cases:
        try:
            got = sol.predict_list_capacities(n)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(n):
            ok = 0.0
            break
    return {"exact_match": ok}
