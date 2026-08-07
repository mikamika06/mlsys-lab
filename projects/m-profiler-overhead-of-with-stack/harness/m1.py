import ref

def check(workdir):
    from proftune.annotations import annotate_step
    out = {"annotations_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.annotate_step(cfg)
        got = annotate_step(cfg)
        if got == want:
            ok += 1
    out["annotations_matched"] = float(ok)
    return out
