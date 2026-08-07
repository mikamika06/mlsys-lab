import sys

sys.path.insert(0, ".")
from roofbench.calc import compute_gbps, roofline_lower_bound_time_ms, find_roofline_knee


def test_compute_gbps_positive():
    res = compute_gbps(1.0, 1024 * 1024 * 1000)
    assert res > 0


def test_roofline_lower_bound():
    t = roofline_lower_bound_time_ms(1024 * 1024 * 1000, 1000.0)
    assert t > 0


def test_find_roofline_knee_valid():
    bs = [16, 32, 64, 128, 256]
    gbps = [100.0, 500.0, 900.0, 910.0, 912.0]
    knee = find_roofline_knee(bs, gbps)
    assert knee in bs
    assert knee <= 128
