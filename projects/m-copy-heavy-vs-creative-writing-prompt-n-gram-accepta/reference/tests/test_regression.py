from ngrameval.metrics import compute_acceptance_metrics, classify_workload


def test_acceptance_metrics_bounds():
    metrics = compute_acceptance_metrics([(0, 4), (1, 4)], 1.0)
    assert 0.0 <= metrics["acceptance_rate"] <= 1.0
    assert metrics["entropy_gap"] >= 0.0


def test_workload_classification():
    label = classify_workload(0.8, 0.9)
    assert label == "copy-heavy"
    label_creative = classify_workload(0.05, 0.1)
    assert label_creative == "creative-writing"
