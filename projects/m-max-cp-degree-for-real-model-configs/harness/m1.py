import ref


def check(workdir):
    from cpdegree.config import max_cp_degree as learner_fn

    out = {"configs_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.max_cp_degree(cfg)
        got = learner_fn(cfg)
        if got == want:
            ok += 1
    out["configs_matched"] = float(ok)
    return out
