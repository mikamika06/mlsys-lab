import ref


def check(workdir):
    from hpa.sim import simulate_hpa

    out = {"sim_matches": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.simulate_hpa(cfg["load"], cfg["target"], cfg["window"], cfg["min"], cfg["max"])
        got = simulate_hpa(cfg["load"], cfg["target"], cfg["window"], cfg["min"], cfg["max"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["sim_matches"] = float(ok)
    return out
