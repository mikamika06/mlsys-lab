import ref

def check(workdir):
    from serverdiag.parser import parse_logs
    out = {"timelines_matched": 0.0}
    ok = 0
    for case in ref.CASES:
        got = parse_logs(case["logs"])
        want = ref.parse_logs(case["logs"])
        if got == want:
            ok += 1
    out["timelines_matched"] = float(ok)
    return out
