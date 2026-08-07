import random

def reference_flag_outliers(X: list[list[float]], factor: float = 3.0) -> list[bool]:
    rows = len(X)
    cols = len(X[0]) if rows > 0 else 0

    m = []
    for j in range(cols):
        max_val = 0.0
        for i in range(rows):
            val = abs(X[i][j])
            if val > max_val:
                max_val = val
        m.append(max_val)

    sorted_m = sorted(m)
    if cols == 0:
        med = 0.0
    elif cols % 2 == 1:
        med = sorted_m[cols // 2]
    else:
        med = (sorted_m[cols // 2 - 1] + sorted_m[cols // 2]) / 2.0

    threshold = factor * med
    return [val > threshold for val in m]

def grade(sol, fx) -> dict:
    random.seed(42)

    test_cases = [
        ([[0.0, 1.0, 10.0], [2.0, 3.0, 12.0], [4.0, 5.0, 13.0]], 3.0),
        ([[1.0]], 3.0),
        ([[1.0, 2.0]], 1.0),
        ([[-5.0, 0.0], [3.0, -10.0]], 2.0),
    ]

    for _ in range(50):
        rows = random.randint(1, 20)
        cols = random.randint(1, 20)
        X = [[random.uniform(-100.0, 100.0) for _ in range(cols)] for _ in range(rows)]
        factor = random.choice([1.0, 1.5, 2.0, 3.0, 5.0])
        test_cases.append((X, factor))

    matches = 0
    total = len(test_cases)

    for X, factor in test_cases:
        expected = reference_flag_outliers(X, factor)
        try:
            result = sol.flag_outliers(X, factor)
        except Exception:
            result = None

        if isinstance(result, list) and len(result) == len(expected) and all(bool(r) == bool(e) for r, e in zip(result, expected)):
            matches += 1

    exact_match = 1.0 if matches == total else 0.0
    return {"exact_match": exact_match}
