from threadperf.classifier import classify_runs
from threadperf.metrics import compute_performance_metrics


def analyze_runs(runs):
    classifications = classify_runs(runs)
    metrics = compute_performance_metrics(runs)
    return {"classifications": classifications, "metrics": metrics}
