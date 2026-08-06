import ref

def check(workdir):
    from leak.drift import detect_drift
    cases = ref.get_test_cases()
    res1 = detect_drift(cases[0])
    res2 = detect_drift(cases[1])
    want1 = {"has_drift": True, "slope": res1["slope"]}
    want2 = {"has_drift": False, "slope": res2["slope"]}
    ok = (res1["has_drift"] == want1["has_drift"]) and (res2["has_drift"] == want2["has_drift"])
    return {"drift_matched": 1.0 if ok else 0.0}
