import ref

def check(workdir):
    from triton_verify.scaling import compute_scaling_efficiency
    out = {"efficiency_matched": 0.0}
    throughputs = [100.0, 190.0, 350.0]
    want = ref.compute_scaling_efficiency(ref.CONFIGS, throughputs)
    got = compute_scaling_efficiency(ref.CONFIGS, throughputs)
    if got == want:
        out["efficiency_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
