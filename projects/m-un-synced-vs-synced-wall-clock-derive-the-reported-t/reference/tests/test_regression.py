import sys

sys.path.insert(0, ".")
from timing.derive import derive_reported_time_gap
from timing.trace import find_missing_cuda_synchronize
from timing.mps import measure_mps_synchronize_cost


def test_derive_reported_time_gap_non_negative():
    val = derive_reported_time_gap([1.0, 1.0], [5.0, 5.0])
    assert val >= 0.0


def test_find_missing_cuda_synchronize_valid():
    events = [{"type": "gpu_idle", "duration": 200, "threshold": 50, "has_sync": False}]
    assert find_missing_cuda_synchronize(events) == 0


def test_measure_mps_synchronize_cost_positive():
    cost = measure_mps_synchronize_cost([10.0, 12.0], [5.0, 6.0])
    assert cost > 0.0
