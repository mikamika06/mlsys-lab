import ref

def check(workdir):
    from engineprof.inspector import count_precisions
    out = {"counts_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.count_precisions(cfg)
        got = count_precisions(cfg)
        if got == want:
            ok += 1
    out["counts_matched"] = float(ok)
    return out
