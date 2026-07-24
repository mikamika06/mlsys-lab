def _broadcast_shape_ref(a, b):
    """Compute the NumPy-style broadcast result shape (deterministic oracle)."""
    if len(a) > len(b):
        b = (1,) * (len(a) - len(b)) + b
    elif len(b) > len(a):
        a = (1,) * (len(b) - len(a)) + a
    result = []
    for x, y in zip(a, b):
        if x == y:
            result.append(x)
        elif x == 1:
            result.append(y)
        elif y == 1:
            result.append(x)
        else:
            return "incompatible"
    return tuple(result)

def grade(sol, fx) -> dict:
    test_cases = [
        ((3, 4), (3, 4)),
        ((3, 1), (1, 4)),
        ((3, 4), (4,)),
        ((5, 3, 4), (3, 4)),
        ((5, 3, 4), (4, 1)),
        ((3, 4), (5, 3, 4)),
        ((1,), (5,)),
        ((3, 1), (1, 5, 1)),
        ((2, 3, 4), (5, 3)),
        ((6,), (3,)),
        ((2, 1, 4), (3, 1)),
        ((1, 2), (3,)),
        ((2,), (3,)),
        ((2, 3), (4, 5)),
        ((), (3,)),
        ((), ()),
        ((1, 1, 1), (5,)),
        ((5, 1, 3), (1, 4, 1)),
        ((3, 1, 5), (1, 4, 1)),
        ((2, 3), (2, 3)),
    ]

    correct = 0
    total = len(test_cases)
    for a, b in test_cases:
        ref = _broadcast_shape_ref(a, b)
        try:
            got = sol.broadcast_shape(a, b)
        except Exception:
            got = None
        if got == ref:
            correct += 1

    return {"exact_match": correct / total if total > 0 else 0.0}
