import ref


def check(workdir):
    from quantres.resolver import resolve_targets

    ok = 0
    out = {"resolvers_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.resolve_targets(cfg["modules"], cfg["targets"], cfg["ignores"])
        got = resolve_targets(cfg["modules"], cfg["targets"], cfg["ignores"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["resolvers_matched"] = float(ok)
    return out
