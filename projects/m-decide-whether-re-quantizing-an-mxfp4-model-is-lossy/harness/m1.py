import ref


def check(workdir):
    from mxfp4.analysis import is_requantization_lossy

    out = {"decisions_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = True
        got = is_requantization_lossy(cfg)
        if got == want:
            ok += 1
    out["decisions_matched"] = float(ok)
    return out
