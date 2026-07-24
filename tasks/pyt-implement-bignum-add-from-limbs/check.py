import inspect


BASE = 1 << 30


def _to_int(limbs):
    value = 0
    for i, limb in enumerate(limbs):
        value += limb * (BASE ** i)
    return value


def _from_int(value):
    if value == 0:
        return [0]
    out = []
    while value:
        out.append(value & (BASE - 1))
        value >>= 30
    return out


def _oracle(a, b):
    return _from_int(_to_int(a) + _to_int(b))


def grade(sol, fx) -> dict:
    cases = [
        ([0], [0]),
        ([1], [2]),
        ([BASE - 1], [1]),
        ([BASE - 1, 3], [1, 7]),
        ([12, 0, 4, 9], [BASE - 1, BASE - 1, 2]),
        ([BASE - 1] * 6, [1]),
        ([123456, 789012, 345678], [987654, 111111]),
        ([0, 0, 1], [BASE - 1, BASE - 1]),
    ]

    exact_match = 1.0

    try:
        source = inspect.getsource(sol.add_limbs)
        normalized = "".join(source.split())
        forbidden = [
            "x+y",
            "y+x",
            "value=x+y",
            "value=y+x",
            "returnx+y",
            "returny+x",
        ]
        if any(pattern in normalized for pattern in forbidden):
            exact_match = 0.0
    except Exception:
        exact_match = 0.0

    if exact_match:
        for a, b in cases:
            try:
                got = list(sol.add_limbs(list(a), list(b)))
            except Exception:
                exact_match = 0.0
                break
            if got != _oracle(a, b):
                exact_match = 0.0
                break

    return {"exact_match": exact_match}
