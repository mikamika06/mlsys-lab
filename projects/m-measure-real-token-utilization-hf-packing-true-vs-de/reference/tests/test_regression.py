import sys
sys.path.insert(0, ".")
from packutil.analyzer import measure_utilization, detect_leak, evaluate_costs


def test_measure_utilization_bounds():
    res = measure_utilization([100, 200, 150], 512)
    assert 0.0 < res["padded_utilization"] <= 1.0
    assert res["packed_utilization"] >= res["padded_utilization"]


def test_detect_leak_correctness():
    pos = [0, 1, 2, 0, 1, 2]
    boundaries = [3]
    assert detect_leak(pos, boundaries) is False
    bad_pos = [0, 1, 2, 3, 4, 5]
    assert detect_leak(bad_pos, boundaries) is True


def test_evaluate_costs_savings():
    costs = evaluate_costs([200, 200, 200], 512, 768)
    assert costs["packed_bytes"] <= costs["padded_bytes"]
