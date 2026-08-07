import sys
sys.path.insert(0, ".")
from trace.replay import replay_trace
from trace.metrics import compute_occupancy

def test_timeline_non_negative():
    trace = [{"start": 0, "duration": 5, "blocks": 4}]
    timeline = replay_trace(trace)
    assert all(x >= 0 for x in timeline)

def test_peak_greater_or_equal_mean():
    trace = [{"start": 0, "duration": 4, "blocks": 2}, {"start": 2, "duration": 3, "blocks": 3}]
    timeline = replay_trace(trace)
    m = compute_occupancy(timeline)
    assert m["peak"] >= m["mean"]

def test_empty_trace():
    timeline = replay_trace([])
    m = compute_occupancy(timeline)
    assert m["peak"] == 0
    assert m["mean"] == 0.0
