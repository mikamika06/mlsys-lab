import sys

sys.path.insert(0, ".")
from moe_sweep.bandwidth import derive_bandwidth


def test_bandwidth_positive():
    bw = derive_bandwidth(100.0, 1024 * 1024)
    assert bw > 0


def test_bandwidth_scaling():
    bw1 = derive_bandwidth(100.0, 1000)
    bw2 = derive_bandwidth(200.0, 1000)
    assert bw2 == bw1 * 2
