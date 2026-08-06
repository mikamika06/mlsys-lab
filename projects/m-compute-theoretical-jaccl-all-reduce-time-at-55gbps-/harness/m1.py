import ref


def check(workdir):
    from jaccl.perf import compute_allreduce_time

    errs = []
    for tc in ref.TEST_CASES_M1:
        want = ref.ref_allreduce_time(**tc)
        got = compute_allreduce_time(**tc)
        errs.append(abs(got - want) / abs(want))
    max_err = max(errs)
    return {"rel_err": float(max_err)}
