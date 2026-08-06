import ref


def check(workdir):
    from quant_target import targeting
    out = {"targets_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.filter_target_modules(cfg)
        got = targeting.filter_target_modules(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["targets_matched"] = float(ok)
    return out
