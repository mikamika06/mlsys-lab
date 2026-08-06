import ref

def check(workdir):
    from spec.capture import capture_acceptance_rate
    out = {"metrics_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.capture_acceptance_rate(cfg, 4)
        got = capture_acceptance_rate(cfg, 4)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, reference {want}"
    out["metrics_matched"] = float(ok)
    return out
