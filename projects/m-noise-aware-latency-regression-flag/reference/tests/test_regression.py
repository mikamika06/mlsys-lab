import sys
sys.path.insert(0, ".")
from latency.stats import compute_robust_stats
from latency.detector import classify_run
from latency.gating import validate_execution_mode

def test_robust_statistics_filtering():
    samples = [100, 101, 100, 102, 1000]
    stats = compute_robust_stats(samples)
    assert stats["median"] == 101.0
    assert stats["mad"] == 1.0

def test_reassociation_classification():
    samples = [100, 101, 102, 100, 101]
    status = classify_run(samples, baseline_med=100.0, baseline_mad=1.0, max_rel_diff=1e-5)
    assert status in ("reassociation", "normal")

def test_silent_eager_fallback_detection():
    fallback_samples = [250, 255, 248, 252, 251]
    valid = validate_execution_mode(fallback_samples, baseline_med=100.0, baseline_mad=1.0, max_rel_diff=1e-5)
    assert valid is False, "Execution mode validation must fail when silent eager fallback occurs"
