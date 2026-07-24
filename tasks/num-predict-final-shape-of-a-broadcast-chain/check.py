import numpy as np

def _oracle(ops):
    """Use NumPy to compute the true final shape of the broadcast chain."""
    shape = ops[0][1]
    cur = np.zeros(shape)
    for op_name, shape in ops[1:]:
        other = np.zeros(shape)
        if op_name == "add":
            cur = cur + other
        elif op_name == "subtract":
            cur = cur - other
        elif op_name == "multiply":
            cur = cur * other
        elif op_name == "divide":
            cur = cur / other
        else:
            raise ValueError(f"unknown op {op_name!r}")
    return tuple(cur.shape)

def grade(sol, fx) -> dict:
    test_cases = [
        [("init", (3, 1)), ("multiply", (1, 4)), ("add", (2, 1, 1))],
        [("init", (5,)), ("add", (1,)), ("multiply", (5,))],
        [("init", (2, 1, 3)), ("multiply", (1, 4, 1)), ("add", (4, 3))],
        [("init", (1,)), ("add", (3, 1)), ("multiply", (1, 5))],
        [("init", (7, 1)), ("multiply", (1, 1))],
        [("init", (1, 2, 1, 3)), ("multiply", (4, 1, 5, 1))],
        [("init", (1,)), ("add", (1,)), ("multiply", (1,)), ("add", (1,))],
        [("init", (6, 1)), ("add", (1, 7))],
        [
            ("init", (1, 3, 1)),
            ("multiply", (2, 1, 4)),
            ("add", (1, 1)),
            ("multiply", (3, 1, 1, 1)),
        ],
        [("init", (3, 4, 5))],
    ]

    ok = 1.0
    for ops in test_cases:
        expected = _oracle(ops)
        try:
            got = sol.predict_broadcast_shape(ops)
            got = tuple(got)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
