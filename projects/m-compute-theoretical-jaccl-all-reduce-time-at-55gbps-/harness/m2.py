import ref


def check(workdir):
    from jaccl.perf import compute_allreduce_overhead

    errs = []
    for tc in ref.TEST_CASES_M2:
        want = ref.ref_allreduce_overhead(**tc)
        got = compute_allreduce_overhead(**tc)
        errs.append(abs(got - want) / abs(want))
    max_err = max(errs)
    return {"rel_err": float(max_err)}
