import ref


def check(workdir):
    from dsdiag.timeline import compute_overlap_ratio
    out = {"overlap_ratio_match": 0.0}
    matches = 0
    for t in ref.TIMELINES:
        got = compute_overlap_ratio(t)
        want = ref.compute_overlap_ratio(t)
        if abs(got - want) < 1e-5:
            matches += 1
    if matches == len(ref.TIMELINES):
        out["overlap_ratio_match"] = 1.0
    return out
