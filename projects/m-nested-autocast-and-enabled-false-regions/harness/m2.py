import ref


def check(workdir):
    from nested_cast.context import resolve_effective_states
    out = {"effective_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.resolve_states(cfg)
        got = resolve_effective_states(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, want {want}"
    if ok == len(ref.CONFIGS):
        out["effective_matched"] = 1.0
    return out
