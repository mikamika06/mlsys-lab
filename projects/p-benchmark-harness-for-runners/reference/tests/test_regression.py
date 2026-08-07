import sys

sys.path.insert(0, ".")
from runner.core import check_consistency, measure_latency, required_samples


def test_measure_latency_basic():
    res = measure_latency([10, 20, 20, 20, 20], warmup_count=1)
    assert res["median"] == 20.0


def test_required_samples_positive():
    n = required_samples(std_dev=1.0, target_width=0.5)
    assert n > 0


def test_check_consistency_overlapping():
    intervals = [(10, 20), (12, 22), (11, 21)]
    assert check_consistency(intervals) is True
