import ref


def check(workdir):
    from tf32guard.error import compute_relative_error

    cases = ref.get_test_cases()
    ok = 0
    for a, b in cases:
        want = ref.reference_error(a, b)
        try:
            got = float(compute_relative_error(a, b))
        except Exception:
            got = -1.0
        if abs(got - want) < 1e-7:
            ok += 1
    return {"error_matched": float(ok)}
