import ref

def check(workdir):
    from sdpa_exp.kernels import get_kernel_priority
    out = {"priority_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.get_kernel_priority(cfg)
        got = get_kernel_priority(cfg)
        if got == want:
            ok += 1
    out["priority_matched"] = float(ok)
    return out
