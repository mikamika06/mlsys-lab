import ref


def check(workdir):
    from weightloader.loader import simulate_load

    out = {"footprint_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        from reference.weightloader.loader import simulate_load as ref_load
        got = simulate_load(cfg, "heap")
        want = ref_load(cfg, "heap")
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["footprint_matched"] = float(ok)
    return out
