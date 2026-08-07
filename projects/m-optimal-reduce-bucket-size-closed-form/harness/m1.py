import ref

def check(workdir):
    from zero_diag.parser import parse_memory_estimator_log

    out = {"logs_parsed": 0.0}
    ok = 0
    total = len(ref.LOG_FIXTURES)

    for log_str, expected in ref.LOG_FIXTURES:
        got = parse_memory_estimator_log(log_str)
        if got == expected:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"Expected {expected}, got {got}"

    if ok == total:
        out["logs_parsed"] = 1.0
    return out
