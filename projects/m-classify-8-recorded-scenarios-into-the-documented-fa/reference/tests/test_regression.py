from specfail.classifier import classify_scenarios, TAXONOMY
from specfail.metrics import compute_metrics

def test_classification_invariants():
    scenarios = [
        {"id": i, "features": {
            "accept_rate": 0.01 if i == 0 else 0.8,
            "logit_diff": 6.0 if i == 1 else 0.0,
            "cache_err": 1 if i == 2 else 0,
            "rollback_fault": 1 if i == 3 else 0,
            "sync_stall": 1 if i == 4 else 0,
            "mem_leak": 1 if i == 5 else 0,
            "num_drift": 0.05 if i == 6 else 0.0,
            "latency_spike": 3.0 if i == 7 else 0.0
        }} for i in range(8)
    ]
    res = classify_scenarios(scenarios)
    assert len(res) == 8
    categories = [r["category"] for r in res]
    for cat in categories:
        assert cat in TAXONOMY
    # Enforce strict invariant: ensure every category in TAXONOMY is covered at least once by this test set
    for t in TAXONOMY:
        assert t in categories, f"Taxonomy category {t} not covered in regression test scenarios"
    metrics = compute_metrics(res)
    assert "confidence_score" in metrics
    assert metrics["confidence_score"] == 1.0
