import ref


def check(workdir):
    from tritondrain.timeout import derive_minimum_drain_timeout

    configs = ref.CONFIGS + ref.generate_synthetic_configs(seed=99, count=10)
    out = {"timeouts_matched": 0.0, "total_configs": float(len(configs))}
    ok = 0

    for i, cfg in enumerate(configs):
        want = ref.derive_minimum_drain_timeout(cfg)
        got = derive_minimum_drain_timeout(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"

    if ok == len(configs):
        out["timeouts_matched"] = 1.0
    return out
