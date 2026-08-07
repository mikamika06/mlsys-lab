import ref

def check(workdir):
    from bnbdiag.classifier import classify_traceback
    out = {"classification_accuracy": 0.0}
    ok = 0
    total = len(ref.CLASSIFICATIONS)
    for parsed, expected in ref.CLASSIFICATIONS:
        got = classify_traceback(parsed)
        if got == expected:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"classification got {got}, expected {expected}"
    out["classification_accuracy"] = float(ok) / float(total) if total > 0 else 0.0
    return out
