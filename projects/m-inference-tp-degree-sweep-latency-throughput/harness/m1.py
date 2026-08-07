import ref


def check(workdir):
    from inference.tp_sweep import sweep_tp_performance

    out = {"sweeps_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        tps = ref.TP_DEGREES_LIST[i]
        want = ref.sweep_tp_performance(cfg, tps)
        got = sweep_tp_performance(cfg, tps)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["sweeps_matched"] = float(ok)
    return out
