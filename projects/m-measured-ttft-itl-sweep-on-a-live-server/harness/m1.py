import ref

def check(workdir):
    from sweep.server import simulate_sweep
    out = {"sweep_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.simulate_sweep(cfg)
        got = simulate_sweep(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["sweep_matched"] = float(ok)
    return out
