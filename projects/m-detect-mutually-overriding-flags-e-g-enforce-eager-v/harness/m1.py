import ref


def check(workdir):
    from vllmargs.conflict import detect_conflicts

    out = {"conflicts_matched": 0.0, "total": float(len(ref.CONFLICT_CASES))}
    ok = 0
    for i, (args, want) in enumerate(ref.CONFLICT_CASES):
        got = detect_conflicts(args)
        if sorted(got) == sorted(want):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    out["conflicts_matched"] = float(ok)
    return out
