import ref


def check(workdir):
    from spec.measure import measure_speedup

    out = {"speedup_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        base = cfg["tokens_per_sec_base"]
        rate = cfg["accept_rates"][4]
        want = ref.compute_speedup(base, base * 2.5, rate, 4)
        got = measure_speedup(cfg, num_draft_tokens=4, override_rate=rate)
        if abs(got - want) < 1e-2:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["speedup_matched"] = float(ok)
    return out
