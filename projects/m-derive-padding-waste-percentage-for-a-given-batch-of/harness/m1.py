import ref


def check(workdir):
    from packeff.waste import compute_waste_percentage
    cases = ref.get_test_cases()
    ok = 0
    total = len(cases)
    for lengths, max_len in cases:
        want = ref.compute_waste_percentage(lengths, max_len)
        try:
            got = compute_waste_percentage(lengths, max_len)
            if abs(got - want) < 1e-5:
                ok += 1
        except Exception:
            pass
    return {"waste_matched": 1.0 if ok == total else 0.0}
