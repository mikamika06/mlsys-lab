import ref

def check(workdir):
    from triton_verify.parser import compute_concurrency_ceiling
    out = {"ceilings_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.compute_concurrency_ceiling(cfg)
        got = compute_concurrency_ceiling(cfg)
        if got == want:
            ok += 1
    out["ceilings_matched"] = float(ok)
    return out
