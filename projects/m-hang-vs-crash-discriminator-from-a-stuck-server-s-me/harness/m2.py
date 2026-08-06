import ref

def check(workdir):
    from serverdiag.detector import classify_failure
    out = {"classification_accuracy": 0.0}
    correct = 0
    for case in ref.CASES:
        got = classify_failure(case["logs"], case["metrics"])
        want = case["expected"]
        if got == want:
            correct += 1
    out["classification_accuracy"] = float(correct) / len(ref.CASES) if ref.CASES else 0.0
    return out
