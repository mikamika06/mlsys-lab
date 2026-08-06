import ref


def check(workdir):
    from mpsmem.fraction import enforce_fraction, check_oom
    out = {"fraction_matched": 0.0, "oom_caught": 0.0}
    ok_frac = 0
    ok_oom = 0
    for cap in ref.TEST_CAPACITIES:
        want_limit = ref.enforce_fraction(0.5, cap)
        got_limit = enforce_fraction(0.5, cap)
        if want_limit == got_limit:
            ok_frac += 1

    if ok_frac == len(ref.TEST_CAPACITIES):
        out["fraction_matched"] = 1.0

    oom1 = ref.check_oom(500, 450, 100)
    got_oom1 = check_oom(500, 450, 100)
    oom2 = ref.check_oom(500, 100, 100)
    got_oom2 = check_oom(500, 100, 100)
    if oom1 == got_oom1 and oom2 == got_oom2:
        out["oom_caught"] = 1.0
    return out
