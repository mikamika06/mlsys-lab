import ref


def check(workdir):
    from probe import head_dim

    out = {"ceilings_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.check_head_ceiling(cfg)
        got = head_dim.check_head_ceiling(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["ceilings_matched"] = float(ok)
    return out
