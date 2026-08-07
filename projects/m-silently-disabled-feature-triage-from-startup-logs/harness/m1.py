import ref

def check(workdir):
    from triage.analyzer import parse_startup_logs
    out = {"features_matched": 0.0}
    ok = 0
    for logs, expected in ref.LOGS_TESTS:
        got = parse_startup_logs(logs)
        if sorted(got) == sorted(expected):
            ok += 1
    out["features_matched"] = float(ok)
    return out
