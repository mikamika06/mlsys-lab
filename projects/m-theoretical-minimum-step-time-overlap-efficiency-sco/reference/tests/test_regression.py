"""Regression tests for overlap profiling metrics."""

import sys

sys.path.insert(0, ".")
from overlap.metrics import compute_metrics
from overlap.reconstruct import reconstruct_timeline
from overlap.saturation import find_saturation_point


def test_reconstruction_invariants():
    events = [
        {"id": 1, "kind": "compute", "start": 0.0, "end": 10.0, "stream": 0},
        {"id": 2, "kind": "comm", "start": 5.0, "end": 15.0, "stream": 1},
    ]
    recon = reconstruct_timeline(events)
    assert recon["total_span"] == 15.0
    assert recon["compute_only"] == 5.0
    assert recon["overlapped"] == 5.0
    assert recon["comm_only"] == 5.0
    assert recon["idle"] == 0.0


def test_metrics_theoretical_minimum():
    events = [
        {"id": 1, "kind": "compute", "start": 0.0, "end": 20.0, "stream": 0},
        {"id": 2, "kind": "comm", "start": 10.0, "end": 30.0, "stream": 1},
    ]
    m = compute_metrics(events)
    assert m["theoretical_min_step_time"] == 20.0
    assert m["actual_step_time"] == 30.0
    assert m["overlap_efficiency_score"] == 0.5


def test_saturation_point_selection():
    profiles = [
        {"bucket_size_mb": 1, "total_step_time": 120.0},
        {"bucket_size_mb": 4, "total_step_time": 95.0},
        {"bucket_size_mb": 16, "total_step_time": 80.0},
        {"bucket_size_mb": 64, "total_step_time": 105.0},
    ]
    res = find_saturation_point(profiles)
    assert res["saturation_bucket_mb"] == 16
    assert res["min_step_time"] == 80.0
