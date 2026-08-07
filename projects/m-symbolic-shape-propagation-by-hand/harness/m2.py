import ref


def check(workdir):
    from symshape.infer import find_first_failure
    from symshape.coverage import compute_coverage

    out = {"first_failures_matched": 0.0, "coverage_matched": 0.0}

    fail_ok = 0
    cov_ok = 0
    total = len(ref.GRAPHS)

    for i, graph in enumerate(ref.GRAPHS):
        want_fail = ref.find_first_failure(graph)
        got_fail = find_first_failure(graph)
        if got_fail == want_fail:
            fail_ok += 1
        elif "_note" not in out:
            out["_note"] = f"graph {i} failure node: got {got_fail}, expected {want_fail}"

        want_cov = ref.compute_coverage(graph)
        got_cov = compute_coverage(graph)
        if got_cov == want_cov:
            cov_ok += 1
        elif "_note" not in out:
            out["_note"] = f"graph {i} coverage: got {got_cov}, expected {want_cov}"

    if fail_ok == total:
        out["first_failures_matched"] = 1.0
    if cov_ok == total:
        out["coverage_matched"] = 1.0

    return out
