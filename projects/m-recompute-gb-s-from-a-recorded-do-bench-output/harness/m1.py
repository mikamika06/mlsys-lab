import ref


def check(workdir):
    from roofbench.calc import compute_gbps
    max_rel_err = 0.0
    for tc in ref.TEST_CASES_M1:
        want = ref.compute_gbps(tc["ms"], tc["bytes"])
        got = compute_gbps(tc["ms"], tc["bytes"])
        err = abs(got - want) / (abs(want) + 1e-12)
        if err > max_rel_err:
            max_rel_err = err
    return {"rel_err_gbps": float(max_rel_err)}
