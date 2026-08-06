import ref


def check(workdir):
    from ttft.savings import prefill_cost

    max_err = 0.0
    for args in ref.M1_CASES:
        want = ref.prefill_cost(*args)
        got = prefill_cost(*args)
        err = abs(want - got) / (abs(want) + 1e-9)
        if err > max_err:
            max_err = err

    return {"rel_err": max_err}
