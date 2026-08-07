import ref

def check(workdir):
    from tritonfix.detect import detect_mismatch
    out = {"mismatches_detected": 0.0}
    ok = 0
    for case in ref.MISMATCH_CASES:
        want = ref.detect_mismatch(case["config"], case["files"])
        got = detect_mismatch(case["config"], case["files"])
        if got == want:
            ok += 1
    out["mismatches_detected"] = float(ok)
    return out
