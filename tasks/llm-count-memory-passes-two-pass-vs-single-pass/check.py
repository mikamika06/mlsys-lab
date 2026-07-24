def _reference_counts(N, D):
    # Two‑pass reads each element twice.
    two_pass = 2 * N * D
    # Single‑pass reads each element once.
    single_pass = N * D
    return two_pass, single_pass


def grade(sol, fx) -> dict:
    cases = [
        (10, 5),
        (100, 20),
        (1, 1),
        (7, 13),
        (256, 64)
    ]
    ok = 1.0
    for N, D in cases:
        try:
            got = sol.count_memory_passes(N, D)
            if not isinstance(got, tuple) or len(got) != 2:
                ok = 0.0
                break
            two, single = got
            ref_two, ref_single = _reference_counts(N, D)
            if two != ref_two or single != ref_single:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
