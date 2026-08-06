import ref


def check(workdir):
    from flashdiag import triage

    out = {"matches_count": 0.0}
    cases = ref.TRIAGE_CASES
    ok = 0
    for i, case in enumerate(cases):
        msg = case["error_msg"]
        res = triage.lookup_error(msg)
        if res and res.get("cause") == case["cause"] and res.get("fix") == case["fix"]:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {res}, expected cause={case['cause']}, fix={case['fix']}"
    out["matches_count"] = float(ok)
    return out
