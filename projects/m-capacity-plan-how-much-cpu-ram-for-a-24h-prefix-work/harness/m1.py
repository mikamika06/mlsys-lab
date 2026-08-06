import ref

def check(workdir):
    from capacity.plan import calculate_prefix_ram_bytes
    out = {"plans_matched": 0.0, "total": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.build_plan(cfg["model_cfg"], cfg["prefixes"], cfg["retention_hours"])
        got = calculate_prefix_ram_bytes(cfg["model_cfg"], cfg["prefixes"], cfg["retention_hours"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"cfg {i}: got {got}, want {want}"
    out["plans_matched"] = float(ok)
    return out
