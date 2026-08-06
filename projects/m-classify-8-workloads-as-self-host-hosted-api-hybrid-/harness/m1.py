import ref

def check(workdir):
    from workloads.classifier import classify_all
    try:
        got = classify_all(ref.WORKLOADS)
    except Exception as e:
        return {"classification_matched": 0.0, "_note": f"Exception raised: {type(e).__name__}: {str(e)[:100]}"}
    
    want = ref.get_reference_classifications()
    if not isinstance(got, dict):
        return {"classification_matched": 0.0, "_note": "classify_all must return a dict mapping workload id to dict"}
    
    match = 1.0
    for wid, expected in want.items():
        actual = got.get(wid)
        if actual != expected:
            match = 0.0
            return {"classification_matched": 0.0, "_note": f"Workload {wid}: got {actual}, want {expected}"}
    return {"classification_matched": match}
