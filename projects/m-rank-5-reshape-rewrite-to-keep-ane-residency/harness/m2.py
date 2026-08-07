import ref


def check(workdir):
    from anerewrite.score import compute_residency_score
    out = {"scores_matched": 0.0}
    ok = 0
    for case in ref.PLAN_CASES:
        want = ref.compute_residency_score(case)
        got = compute_residency_score(case)
        if abs(got - want) < 1e-5:
            ok += 1
    if ok == len(ref.PLAN_CASES):
        out["scores_matched"] = 1.0
    return out
