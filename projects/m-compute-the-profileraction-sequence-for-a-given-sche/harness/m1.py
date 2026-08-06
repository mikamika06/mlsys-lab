import ref

def check(workdir):
    from profiler_util.schedule import compute_actions

    out = {"sequences_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_reference_actions(cfg)
        try:
            got = compute_actions(
                cfg["wait"],
                cfg["warmup"],
                cfg["active"],
                cfg["repeat"],
                cfg["total_steps"]
            )
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} raised exception: {type(e).__name__}"
            continue

        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"

    out["sequences_matched"] = float(ok)
    return out
