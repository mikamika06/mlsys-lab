import ref


def check(workdir):
    from timing.derive import derive_reported_time_gap
    cases = ref.get_test_cases_m1()
    max_rel_err = 0.0
    for launches, kernels, want in cases:
        got = derive_reported_time_gap(launches, kernels)
        if want == 0.0:
            err = abs(got - want)
        else:
            err = abs(got - want) / abs(want)
        if err > max_rel_err:
            max_rel_err = err
    out = {"rel_err": float(max_rel_err)}
    if max_rel_err > 0.01:
        out["_note"] = f"max relative error {max_rel_err} exceeds 0.01"
    return out
