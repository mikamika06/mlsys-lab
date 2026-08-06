import ref

def check(workdir):
    from compile_peft.guards import classifies_guard_failure
    out = {"classification_matched": 0.0}
    ok = 0
    for item in ref.GUARDS_TESTS:
        change = item["change"]
        want = item["triggers_guard"]
        got = classifies_guard_failure(change)
        if got == want:
            ok += 1
    out["classification_matched"] = 1.0 if ok == len(ref.GUARDS_TESTS) else 0.0
    return out
