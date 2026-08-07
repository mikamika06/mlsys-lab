import ref


def check(workdir):
    from unfoldfix.trace import minimal_source_trace
    out = {"traces_matched": 0.0}
    ok = 0
    for case in ref.TRACE_TESTS:
        got = minimal_source_trace(case["tb"], case["user_prefix"])
        want = case["want"]
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"trace mismatch: got {got}, want {want}"
    out["traces_matched"] = float(ok)
    return out
