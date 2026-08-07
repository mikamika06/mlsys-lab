import random

def grade(sol, fx) -> dict:
    test_cases = [
        ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], 2),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0], 2),
        ([1.0, 2.0], [3.0, 4.0], 5),
        ([], [], 4),
        ([random.random() for _ in range(100)], [random.random() for _ in range(100)], 16),
        ([random.random() for _ in range(1000)], [random.random() for _ in range(1000)], 64),
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 1),
        ([10.0], [20.0], 1),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], 3),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2),
    ]

    max_err = 0.0
    for a, b, block_size in test_cases:
        expected = [x + y for x, y in zip(a, b)]
        actual = sol.emulated_triton_add(list(a), list(b), block_size)

        if not isinstance(actual, list):
            return {"max_abs_err": float("inf")}

        if len(actual) != len(expected):
            return {"max_abs_err": float("inf")}

        for ex, ac in zip(expected, actual):
            err = abs(ex - ac)
            if err > max_err:
                max_err = err

    return {"max_abs_err": max_err}
