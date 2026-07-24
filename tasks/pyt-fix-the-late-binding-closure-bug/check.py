def grade(sol, fx) -> dict:
    """Grade the make_multipliers function against the arithmetic oracle i*x."""
    try:
        multipliers = sol.make_multipliers()
    except Exception:
        return {"exact_match": 0.0}

    if not isinstance(multipliers, list) or len(multipliers) != 5:
        return {"exact_match": 0.0}

    for i, f in enumerate(multipliers):
        if not callable(f):
            return {"exact_match": 0.0}

    test_values = [0, 1, -1, 7, 42, -13, 100, 256, -1000]

    for i, f in enumerate(multipliers):
        for x in test_values:
            try:
                got = f(x)
                expected = i * x          # oracle — computed, never hardcoded
            except Exception:
                return {"exact_match": 0.0}
            if got != expected:
                return {"exact_match": 0.0}

    return {"exact_match": 1.0}
