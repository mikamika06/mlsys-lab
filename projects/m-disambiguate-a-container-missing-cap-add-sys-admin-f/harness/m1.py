import ref

def check(workdir):
    from profilediag.classifier import classify_error
    out = {"cases_matched": 0.0, "total_cases": float(len(ref.TEST_CASES))}
    ok = 0
    for i, case in enumerate(ref.TEST_CASES):
        got = classify_error(case["log"], case["env"])
        want = case["expected"]
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i} ({case['id']}): got {got}, want {want}"
    out["cases_matched"] = float(ok)
    return out
