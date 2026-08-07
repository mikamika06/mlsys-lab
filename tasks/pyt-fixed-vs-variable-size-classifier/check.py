import sys

def _reference_is_variable(obj):
    t = type(obj)
    try:
        empty = t()
    except Exception:
        return False
    return sys.getsizeof(obj) > sys.getsizeof(empty)

def grade(sol, fx) -> dict:
    fixtures = [
        [], {}, set(), (), "", b"", bytearray(),
        [0], {"a":1}, {0}, (0,), "x", b"x", bytearray(b"x"),
        0, 1, -1, 3.14, True, False, None,
        0+0j, 1+2j,
        memoryview(b"abc"),
        [i for i in range(10)],
        {"key": "value"},
        set(range(5)),
        tuple(range(3)),
        bytearray(b"abcde"),
        [0]*50
    ]
    try:
        got = sol.classify_objects(fixtures)
    except Exception:
        return {"exact_match": 0.0}
    if not isinstance(got, list) or any(not isinstance(x, bool) for x in got):
        return {"exact_match": 0.0}
    ref = [_reference_is_variable(o) for o in fixtures]
    ok = float(got == ref)
    return {"exact_match": ok}
