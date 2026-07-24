import numpy as np

def grade(sol, fx) -> dict:
    # Test cases: pairs of shapes that are broadcastable or not
    cases = [
        ((3, 1), (2, 4)),
        ((5,), (1, 5, 1)),
        ((2, 3), (4,)),
        ((0, 2), (2,)),
        ((7, 1, 5), (1, 5)),
        ((4, 6, 8), (4, 1, 8)),
        ((9,), (9, 9)),
        ((3, 4, 5), (1, 4, 1)),
    ]
    ok = 1.0
    for shape1, shape2 in cases:
        try:
            expected = tuple(np.broadcast_shapes(shape1, shape2))
        except ValueError:
            expected = ()
        try:
            got = sol.broadcast_shape(shape1, shape2)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
