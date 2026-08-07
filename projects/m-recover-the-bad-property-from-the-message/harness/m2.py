import ref

def check(workdir):
    from triage.dtype import find_dtype_leaks
    out = {"dtype_match": 0.0}
    ok = 0
    for state in ref.STATES:
        want = ref.find_dtype_leaks(state)
        got = find_dtype_leaks(state)
        if got == want:
            ok += 1
    if ok == len(ref.STATES):
        out["dtype_match"] = 1.0
    return out
