import sys


def _oracle(attrs):
    names = tuple(attrs.keys())

    DictClass = type("DictClass", (), {})
    SlotsClass = type(
        "SlotsClass",
        (),
        {"__slots__": names},
    )

    dict_obj = DictClass()
    slots_obj = SlotsClass()

    for key, value in attrs.items():
        setattr(dict_obj, key, value)
        setattr(slots_obj, key, value)

    dict_size = sys.getsizeof(dict_obj) + sys.getsizeof(dict_obj.__dict__)
    slots_size = sys.getsizeof(slots_obj)

    return float(dict_size / slots_size)


def grade(sol, fx) -> dict:
    cases = [
        {"x": 1, "y": 2, "name": "sample"},
        {"a": 10, "b": 20, "c": 30, "d": 40},
        {"value": 3.14},
    ]

    ok = 1.0
    for attrs in cases:
        try:
            got = float(sol.instance_footprint_ratio(dict(attrs)))
            ref = _oracle(dict(attrs))
        except Exception:
            ok = 0.0
            break

        if abs(got - ref) > 1e-12:
            ok = 0.0
            break

    return {"size_ratio": ok}
