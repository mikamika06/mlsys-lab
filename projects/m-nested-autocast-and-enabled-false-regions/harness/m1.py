import ref


def check(workdir):
    from nested_cast.core import parse_tree
    out = {"states_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.resolve_states(cfg)
        got = parse_tree(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, want {want}"
    out["states_matched"] = float(ok)
    return out
