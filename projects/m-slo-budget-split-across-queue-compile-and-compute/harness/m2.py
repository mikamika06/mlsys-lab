import ref


def check(workdir):
    from slobudget.optimize import find_optimal_batch

    errs = []
    for c in ref.CASES:
        want = ref.find_optimal_batch(c["slo"], c["compile"], c["per_token"], 32)
        got = find_optimal_batch(c["slo"], c["compile"], c["per_token"], 32)
        if want == 0:
            rel = 0.0 if got == 0 else 1.0
        else:
            rel = abs(got - want) / abs(want)
        errs.append(rel)
    max_err = max(errs) if errs else 1.0
    return {"rel_err": float(max_err)}
