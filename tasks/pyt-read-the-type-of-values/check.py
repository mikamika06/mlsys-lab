def _oracle(values):
    return [type(value).__name__ for value in values]


def grade(sol, fx) -> dict:
    cases = [
        [1, 2.5, "hello", [1, 2]],
        [True, None, (1, 2), {"a": 1}],
        [b"bytes", bytearray(b"x"), {1, 2}, range(3)],
        [type, int, object],
    ]

    ok = 1.0
    for values in cases:
        try:
            got = sol.read_type_names(values)
        except Exception:
            ok = 0.0
            break

        if got != _oracle(values):
            ok = 0.0
            break

    return {"exact_match": ok}
