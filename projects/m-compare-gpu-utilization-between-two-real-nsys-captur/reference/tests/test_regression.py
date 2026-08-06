"""Regression tests for profile analyzer."""

from profile_analyzer.utilization import compute_profile_utilization

def test_stream_overlap_utilization():
    """Verify that overlapping kernels on different streams do not overcount active duration."""
    trace_start = 1000
    trace_end = 2000

    kernels = [
        {"start_ns": 1100, "end_ns": 1500, "stream_id": 1, "device_id": 0},
        {"start_ns": 1200, "end_ns": 1600, "stream_id": 2, "device_id": 0},
    ]

    util = compute_profile_utilization(kernels, trace_start, trace_end)
    expected_active_time = 500
    total_time = 1000
    expected_util = expected_active_time / float(total_time)

    assert abs(util - expected_util) < 1e-6
