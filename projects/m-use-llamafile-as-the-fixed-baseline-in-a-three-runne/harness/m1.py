import ref

def check(workdir):
    from runners.baseline import setup_baseline
    out = {"baseline_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.setup_baseline(cfg)
        got = setup_baseline(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["baseline_matched"] = float(ok)
    return out
