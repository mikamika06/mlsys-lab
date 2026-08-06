import ref


def check(workdir):
    from packutil.analyzer import detect_leak

    test_cases = [
        ([0, 1, 2, 0, 1, 2], [3], False),
        ([0, 1, 2, 3, 4, 5], [3], True),
        ([0, 1, 0, 1, 2], [2], False),
        ([0, 1, 2, 3], [2], True),
    ]
    correct = 0
    total = len(test_cases)
    for pos, bounds, expected in test_cases:
        try:
            res = detect_leak(pos, bounds)
            if bool(res) == bool(expected):
                correct += 1
        except Exception:
            pass
    score = 1.0 if correct == total else 0.0
    out = {"leak_detected": float(score)}
    if score < 1.0:
        out["_note"] = f"passed {correct}/{total} leak detection test cases"
    return out
