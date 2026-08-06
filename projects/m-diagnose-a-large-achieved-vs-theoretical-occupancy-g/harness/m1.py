import ref

def check(workdir):
    from occupancy.resource import rank_kernels
    out = {"rankings_matched": 0.0}
    got = rank_kernels(ref.CONFIGS, ref.LIMITS)
    want = ref.rank_kernels(ref.CONFIGS, ref.LIMITS)
    if got == want:
        out["rankings_matched"] = 1.0
    else:
        out["_note"] = f"got rankings {got}, want {want}"
    return out
