import sys


def _oracle():
    values = [
        0,
        1,
        2**30 - 1,
        2**30,
        2**60 - 1,
        2**60,
        2**90,
    ]
    return [sys.getsizeof(value) for value in values]


def grade(sol, fx) -> dict:
    expected = _oracle()
    try:
        got = sol.int_size_growth()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
