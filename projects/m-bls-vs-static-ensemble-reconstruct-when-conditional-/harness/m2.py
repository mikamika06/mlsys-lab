import ref

def check(workdir):
    from bls_router.metrics import calculate_execution_efficiency

    out = {"bls_efficiency_ratio": 0.0, "static_waste_detected": 0.0}
    
    requests = ref.MOCK_REQUESTS
    metrics = calculate_execution_efficiency(requests)
    
    bls_invocations = metrics.get("bls_invocations", 0)
    ensemble_invocations = metrics.get("static_ensemble_invocations", 0)

    if bls_invocations > 0:
        ratio = ensemble_invocations / bls_invocations
        out["bls_efficiency_ratio"] = float(ratio)

    if ensemble_invocations > bls_invocations:
        out["static_waste_detected"] = 1.0

    return out
