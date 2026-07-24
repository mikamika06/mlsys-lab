def grade(sol, fx) -> dict:
    cases = [
        ([True] * 32, True),
        ([False] * 32, True),
        ([True] * 16 + [False] * 16, False),
        ([1] * 32, True),
        ([0] * 32, True),
        ([1] * 15 + [0] + [1] * 16, False)
    ]
    ok = 1.0
    for pred, expected in cases:
        try:
            got = sol.all_lanes_agree(pred)
        except Exception:
            ok = 0.0
            break
        if bool(got) != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
