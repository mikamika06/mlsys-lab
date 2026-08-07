import ref


def check(workdir):
    from flashfix.audit import triage_warnings

    out = {"warnings_matched": 0.0, "total": float(len(ref.WARNINGS))}
    ok = 0
    for i, w in enumerate(ref.WARNINGS):
        want = ref.triage_warnings([w])
        got = triage_warnings([w])
        if got == want:
            ok += 1
    out["warnings_matched"] = float(ok)
    return out
