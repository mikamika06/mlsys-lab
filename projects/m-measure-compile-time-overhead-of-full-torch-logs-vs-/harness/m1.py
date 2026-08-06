import ref


def check(workdir):
    from tlog.overhead import compute_overhead_ratio
    out = {"ratio_matched": 0.0}
    ok = True
    for tc in ref.TEST_CASES:
        got = compute_overhead_ratio(tc["baseline_time"], tc["logged_time"])
        if abs(got - tc["expected_ratio"]) > 1e-5:
            ok = False
            break
    if ok:
        out["ratio_matched"] = 1.0
    return out
