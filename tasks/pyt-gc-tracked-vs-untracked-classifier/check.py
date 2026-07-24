import gc


def _oracle(objects):
    return [bool(gc.is_tracked(obj)) for obj in objects]


def grade(sol, fx) -> dict:
    cases = [
        None,
        True,
        0,
        1,
        3.14,
        "hello",
        b"bytes",
        (),
        (1, 2, 3),
        ([1, 2],),
        [],
        [1, 2, 3],
        {},
        {"a": 1},
        set(),
        {1, 2, 3},
        object(),
        [object()],
        {"nested": [1, 2]},
        tuple([[], 1]),
    ]

    expected = _oracle(cases)
    try:
        got = sol.classify_gc_tracking(cases)
    except Exception:
        return {"exact_match": 0.0}

    if list(got) == expected:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}
