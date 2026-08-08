import sys
sys.path.insert(0, ".")
from mps_bench.core import inspect_device_flags, measure_execution_time, compare_targets


def test_device_flags_logic():
    res = inspect_device_flags(True, True)
    assert res["valid_state"] is True
    assert res["is_built"] is True


def test_synchronization_penalty():
    trace = [10.0, 10.0, 10.0]
    unsync = measure_execution_time(trace, synchronized=False)
    sync = measure_execution_time(trace, synchronized=True)
    assert unsync < sync


def test_relative_error_bound():
    cpu_vals = [1.0, 2.0, 3.0]
    mps_vals = [1.0001, 2.0001, 3.0001]
    res = compare_targets(cpu_vals, mps_vals)
    assert res["matches_bound"] is True
