def _oracle(pairs):
    return [a is b for a, b in pairs]


def _make_cases():
    literal_a = "identifier_name"
    literal_b = "identifier_name"
    built_a = "".join(["identifier", "_", "name"])
    built_b = ("identifier_" + "name")[:]
    long_literal_a = "another_identifier_123"
    long_literal_b = "another_identifier_123"
    dynamic = "another_" + "identifier_123"
    return [
        (literal_a, literal_b),
        (literal_a, built_a),
        (built_a, built_b),
        (long_literal_a, long_literal_b),
        (long_literal_a, dynamic),
        ("x", "x"),
        ("x", "".join(["x"])),
    ]


def grade(sol, fx) -> dict:
    pairs = _make_cases()
    expected = _oracle(pairs)
    try:
        got = sol.classify_interning(pairs)
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if list(got) == expected else 0.0}
