import sys
sys.path.insert(0, ".")
from mprofiler.bandwidth import compute_bandwidth_percentage
from mprofiler.memory import build_memory_timeline
from mprofiler.signpost import count_kernel_launches

def test_bandwidth_bounds():
    val = compute_bandwidth_percentage(546 * 1e9, 1.0, 546.0)
    assert 99.9 <= val <= 100.1

def test_timeline_non_negative():
    res = build_memory_timeline([100, 200], [150, 250])
    assert all(r["allocated"] >= 0 for r in res)

def test_signpost_parsing():
    lines = ["os_signpost event: kernel_launch", "other line"]
    assert count_kernel_launches(lines) == 1
