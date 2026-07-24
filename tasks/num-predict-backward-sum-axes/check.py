import numpy as np

def _ref(a_shape, b_shape):
    a = np.ones(a_shape)
    b = np.ones(b_shape)
    try:
        c = a * b
    except ValueError:
        return ((0,), (0,))  # or ((0,), (0,))
    
    a_axes = tuple(i for i, s in enumerate(a_shape) if s != 1 and c.shape[i] != s)
    b_axes = tuple(i for i, s in enumerate(b_shape) if s != 1 and c.shape[i] != s)
    return (a_axes, b_axes)


def grade(sol, fx) -> dict:
    cases = [
        ((3, 4), (1, 4)),
        ((2, 3), (3, 1)),
        ((1, 2, 3), (1, 1, 3)),
        ((4, 1, 2), (4, 2, 1)),
    ]
    ok = 1.0
    for a_shape, b_shape in cases:
        try:
            got = sol.predict_backward_sum_axes(a_shape, b_shape)
        except Exception:
            ok = 0.0
            break
        ref = _ref(a_shape, b_shape)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
