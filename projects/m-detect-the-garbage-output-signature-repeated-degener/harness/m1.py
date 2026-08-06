import ref


def check(workdir):
    from engine.detector import detect_garbage

    out = {"signatures_matched": 0.0, "cases": float(len(ref.CASES))}
    ok = 0
    for i, case in enumerate(ref.CASES):
        got = detect_garbage(case["tokens"])
        want = case["expected"]
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    out["signatures_matched"] = float(ok)
    return out
