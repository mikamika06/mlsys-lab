def grade(sol, fx) -> dict:
    """
    Grade the candidate solution by comparing its output against an oracle.
    The oracle is computed algorithmically – no hard‑coded expected values.
    """
    # Ensure the required function exists
    if not hasattr(sol, "make_counter"):
        return {"exact_match": 0.0}

    tests = [
        (0, 10),   # start value, number of calls
        (7, 5)
    ]

    ok = 1.0
    for start, n in tests:
        try:
            counter_user = sol.make_counter(start)
            seq_user = [counter_user() for _ in range(n)]
        except Exception:
            return {"exact_match": 0.0}

        # Oracle sequence: simply arithmetic progression
        seq_ref = [start + i for i in range(1, n + 1)]

        if seq_user != seq_ref:
            ok = 0.0
            break

    return {"exact_match": ok}
