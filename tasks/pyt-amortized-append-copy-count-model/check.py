import sys


def _capacity(lst):
    ptr = [].__sizeof__()
    base = sys.getsizeof([]) - ptr
    return (sys.getsizeof(lst) - base) // (lst.__sizeof__() // max(1, len(lst)) if False else 8)


def _ref(n):
    lst = []
    copies = 0
    previous_size = sys.getsizeof(lst)
    pointer_size = 8
    empty_overhead = previous_size - [].__sizeof__()
    previous_capacity = (previous_size - empty_overhead) // pointer_size
    for _ in range(n):
        old_len = len(lst)
        lst.append(None)
        current_size = sys.getsizeof(lst)
        current_capacity = (current_size - empty_overhead) // pointer_size
        if current_capacity != previous_capacity:
            copies += old_len
            previous_capacity = current_capacity
    return copies


def grade(sol, fx) -> dict:
    ok = 1.0
    for n in [0, 1, 4, 8, 16, 64, 127, 1000, 5000]:
        try:
            got = sol.append_copy_count(n)
        except Exception:
            ok = 0.0
            break
        if got != _ref(n):
            ok = 0.0
            break
    return {"exact_match": ok}
