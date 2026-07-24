import sys


def _reference():
    objects = [
        0,
        0.0,
        tuple(),
        "x",
        b"x",
        True,
        None,
        0j,
    ]
    ranked = sorted(
        objects,
        key=lambda obj: (sys.getsizeof(obj), type(obj).__name__),
    )
    return tuple(type(obj).__name__ for obj in ranked)


def grade(sol, fx) -> dict:
    expected = _reference()
    try:
        got = tuple(sol.rank_object_types())
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
