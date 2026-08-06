import ref


def check(workdir):
    from chooser.briefs import classify_brief
    out = {"briefs_matched": 0.0, "total": float(len(ref.BRIEFS))}
    ok = 0
    for i, b in enumerate(ref.BRIEFS):
        want = ref.classify_brief(b)
        try:
            got = classify_brief(b)
        except Exception:
            got = None
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"brief {i}: got {got}, reference {want}"
    out["briefs_matched"] = float(ok)
    return out
