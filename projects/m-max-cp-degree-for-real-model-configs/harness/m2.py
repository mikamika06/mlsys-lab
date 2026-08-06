import ref


def check(workdir):
    from cpdegree.usp import hybrid_usp_bandwidth as learner_fn

    out = {"usp_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.hybrid_usp_bandwidth(cfg, 900.0, 50.0)
        got = learner_fn(cfg, 900.0, 50.0)
        if abs(got - want) < 1e-5:
            ok += 1
    if ok == len(ref.CONFIGS):
        out["usp_matched"] = 1.0
    return out
