import ref


def check(workdir):
    from isareport.parser import parse_build_log

    out = {"reports_matched": 0.0, "reports": float(len(ref.REPORTS))}
    ok = 0
    for i, rep in enumerate(ref.REPORTS):
        want = ref.parse_log(rep["log"])
        got = parse_build_log(rep["log"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"report {i}: got {got}, reference {want}"
    out["reports_matched"] = float(ok)
    return out
