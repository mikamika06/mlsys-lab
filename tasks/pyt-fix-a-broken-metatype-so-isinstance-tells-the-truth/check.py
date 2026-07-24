def grade(sol, fx) -> dict:
    cases = [
        ("TruthfulInt", int, [0, True, 3.5, "3", None, object()]),
        ("TruthfulText", str, ["x", b"x", 1, None, object()]),
        ("TruthfulContainer", (list, tuple), [[], (), {}, set(), "abc", 4]),
        ("TruthfulNumber", (int, float), [1, 1.5, True, "1", [], object()]),
    ]

    expected = []
    actual = []

    for name, accepted_type, values in cases:
        try:
            T = sol.make_truthful_type(name, accepted_type)
        except Exception:
            return {"exact_match": 0.0}

        for value in values:
            try:
                expected.append(isinstance(value, accepted_type))
                actual.append(isinstance(value, T))
            except Exception:
                return {"exact_match": 0.0}

    return {"exact_match": 1.0 if actual == expected else 0.0}
