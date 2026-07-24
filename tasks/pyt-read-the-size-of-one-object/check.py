import sys


def grade(sol, fx) -> dict:
    cases = [
        0,
        123456789,
        "arena",
        b"bytes",
        [1, 2, 3],
        {"a": 1, "b": 2},
        (1, 2, 3),
        {1, 2, 3},
    ]

    ok = 1.0
    for obj in cases:
        try:
            got = sol.object_size(obj)
            ref = sys.getsizeof(obj)
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, int) or got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
