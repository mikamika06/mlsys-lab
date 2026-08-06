import sys

sys.path.insert(0, ".")
from profiler_utils.metrics import measure_trace_overhead
from profiler_utils.schedule import active_windows_at_step


def test_overhead_ratio_positive():
    traces_no = [b"x" * 100, b"x" * 100]
    traces_with = [b"x" * 250, b"x" * 250]
    ratio = measure_trace_overhead(traces_no, traces_with)
    assert ratio >= 1.0


def test_schedule_monotonicity():
    w1 = active_windows_at_step(10, 2, 2, 4, 0)
    w2 = active_windows_at_step(20, 2, 2, 4, 0)
    assert w2 >= w1


def test_schedule_zero_step():
    w = active_windows_at_step(0, 2, 2, 4, 0)
    assert w == 0 or w == 1
