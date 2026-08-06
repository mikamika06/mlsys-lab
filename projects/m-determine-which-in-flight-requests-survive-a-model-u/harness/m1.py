import ref


def check(workdir):
    from tritondrain.survival import determine_surviving_requests

    configs = ref.CONFIGS + ref.generate_synthetic_configs(seed=42, count=10)
    out = {"survivors_matched": 0.0, "total_configs": float(len(configs))}
    ok = 0

    for i, cfg in enumerate(configs):
        want = ref.determine_surviving_requests(cfg)
        got = determine_surviving_requests(cfg)
        norm_got = sorted(got) if got is not None else []
        if norm_got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {norm_got}, reference {want}"

    if ok == len(configs):
        out["survivors_matched"] = 1.0
    return out
