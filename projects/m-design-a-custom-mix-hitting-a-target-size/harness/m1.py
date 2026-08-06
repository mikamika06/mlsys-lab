import ref

def check(workdir):
    from mixplan.mix import design_mix
    out = {"mix_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.design_mix(cfg["tensors"], cfg["target_bytes"], cfg["options"])
        got = design_mix(cfg["tensors"], cfg["target_bytes"], cfg["options"])
        if got == want:
            ok += 1
    out["mix_matched"] = float(ok)
    return out
