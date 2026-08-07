import ref


def check(workdir):
    from roofbench.calc import roofline_lower_bound_time_ms
    max_rel_err = 0.0
    for tc in ref.TEST_CASES_M2:
        want = ref.roofline_lower_bound_time_ms(tc["bytes"], tc["peak_bw"])
        got = roofline_lower_bound_time_ms(tc["bytes"], tc["peak_bw"])
        err = abs(got - want) / (abs(want) + 1e-12)
        if err > max_rel_err:
            max_rel_err = err
    return {"rel_err_time": float(max_rel_err)}
