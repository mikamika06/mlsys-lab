import ref


def check(workdir):
    from optmem.derive import bytes_per_param

    out = {"bytes_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.bytes_per_param(cfg)
        try:
            got = bytes_per_param(cfg)
        except Exception:
            got = -1
        if got == want:
            ok += 1
    out["bytes_matched"] = float(ok)
    return out
