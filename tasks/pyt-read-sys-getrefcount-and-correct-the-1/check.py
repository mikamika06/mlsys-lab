import sys


def _oracle(obj):
    return sys.getrefcount(obj) - 2


def grade(sol, fx) -> dict:
    cases = [
        [],
        {},
        [1, 2, 3],
        {"a": 1},
        object(),
        "reference-counting",
    ]
    ok = 1.0
    for obj in cases:
        try:
            ref = _oracle(obj)
            got = sol.true_refcount(obj)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
