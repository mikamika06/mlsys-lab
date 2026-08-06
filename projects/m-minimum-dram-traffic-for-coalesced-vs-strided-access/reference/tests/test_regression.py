import sys
sys.path.insert(0, ".")
from coalesce.traffic import min_dram_traffic
from coalesce.simulate import simulate_warp_coalescing
from coalesce.ratio import excess_traffic_ratio


def test_coalesced_traffic_is_minimal():
    t = min_dram_traffic(32, 4, 1)
    assert t == 128


def test_warp_coalescing_count():
    addrs = [i * 4 for i in range(32)]
    assert simulate_warp_coalescing(addrs, 4) == 4


def test_excess_ratio_gt_one_for_strided():
    r = excess_traffic_ratio(32, 4, 2)
    assert r > 1.0
