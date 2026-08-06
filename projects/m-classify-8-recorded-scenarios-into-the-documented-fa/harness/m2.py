import ref

def check(workdir):
    from specfail import classifier, metrics
    scenarios = ref.get_scenarios()
    classifications = classifier.classify_scenarios(scenarios)
    want_metrics = ref.compute_metrics(classifications)
    try:
        got_metrics = metrics.compute_metrics(classifications)
    except Exception as e:
        return {"metrics_match": 0.0, "distribution_valid": 0.0, "_note": f"exception: {e}"}

    metrics_match = 1.0 if got_metrics.get("confidence_score") == want_metrics.get("confidence_score") else 0.0
    
    dist = got_metrics.get("distribution", {})
    total_prob = sum(dist.values())
    distribution_valid = 1.0 if abs(total_prob - 1.0) < 1e-5 else 0.0

    return {
        "metrics_match": metrics_match,
        "distribution_valid": distribution_valid
    }
