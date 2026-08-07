import sys

sys.path.insert(0, ".")
from kvobs.alerting import HysteresisAlert
from kvobs.histogram import calculate_histogram_quantile


def test_histogram_quantile_linear_interpolation():
    buckets = [(0.1, 10.0), (0.5, 50.0), (1.0, 90.0), (2.0, 100.0)]
    q99 = calculate_histogram_quantile(0.99, buckets)
    assert abs(q99 - 1.9) < 1e-4, f"Expected 1.9, got {q99}"


def test_hysteresis_alert_prevents_flapping():
    alert = HysteresisAlert(high_threshold=100.0, low_threshold=50.0, hold_periods=2)
    assert not alert.process(110.0)
    assert alert.process(120.0)
    assert alert.process(80.0)
    assert alert.process(40.0)
    assert not alert.process(30.0)
