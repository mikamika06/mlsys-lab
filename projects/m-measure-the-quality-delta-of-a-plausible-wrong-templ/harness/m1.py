import ref


def check(workdir):
    from templater import quality
    out = {"delta_matched": 0.0}
    ok = 0
    for case in ref.TEST_CASES:
        want = ref.compute_quality_delta(case)
        got = quality.compute_quality_delta(case)
        if abs(got - want) < 1e-5:
            ok += 1
    if ok == len(ref.TEST_CASES):
        out["delta_matched"] = 1.0
    return out
