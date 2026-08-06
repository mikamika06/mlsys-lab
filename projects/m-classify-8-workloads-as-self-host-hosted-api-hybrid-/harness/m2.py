import ref

def check(workdir):
    from workloads.metrics import compute_tco_metrics
    try:
        got = compute_tco_metrics(ref.WORKLOADS)
    except Exception as e:
        return {"metrics_matched": 0.0, "_note": f"Exception raised: {type(e).__name__}: {str(e)[:100]}"}
    
    want = ref.compute_metrics(ref.WORKLOADS)
    if not isinstance(got, dict):
        return {"metrics_matched": 0.0, "_note": "compute_tco_metrics must return a dict"}
    
    match = 1.0
    for wid, expected_val in want.items():
        actual_val = got.get(wid)
        if actual_val != expected_val:
            match = 0.0
            return {"metrics_matched": 0.0, "_note": f"Metrics for {wid}: got {actual_val}, want {expected_val}"}
    return {"metrics_matched": match}
