import ref

def check(workdir):
    from triage.recover import parse_error_property
    out = {"recovered_match": 0.0}
    ok = 0
    for msg in ref.ERROR_MESSAGES:
        want = ref.parse_error_property(msg)
        got = parse_error_property(msg)
        if got == want:
            ok += 1
    if ok == len(ref.ERROR_MESSAGES):
        out["recovered_match"] = 1.0
    return out
