import ref


def check(workdir):
    from zerodp.memory import calc_memory_table

    out = {"memory_tables_matched": 0.0, "total": float(len(ref.CONFIGS_M1))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS_M1):
        want = ref.calc_memory_table(**cfg)
        got = calc_memory_table(**cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"cfg {i}: got {got}, want {want}"
    out["memory_tables_matched"] = 1.0 if ok == len(ref.CONFIGS_M1) else 0.0
    return out
