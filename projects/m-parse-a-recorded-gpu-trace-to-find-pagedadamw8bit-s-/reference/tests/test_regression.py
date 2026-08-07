import sys
sys.path.insert(0, ".")

from gputrace import parse_trace_events, find_spillover_spike


def test_spillover_spike_detection():
    data = {
        "traceEvents": [
            {"name": "PagedAdamW8bit::step", "ts": 100, "dur": 50, "args": {"page_faults": 2, "step": 1}},
            {"name": "PagedAdamW8bit::step", "ts": 200, "dur": 800, "args": {"page_faults": 1, "step": 2}},
            {"name": "PagedAdamW8bit::step", "ts": 300, "dur": 60, "args": {"page_faults": 3, "step": 3}}
        ]
    }
    events = parse_trace_events(data)
    assert len(events) == 3
    res = find_spillover_spike(events)
    assert res["argmin_index"] == 1
    assert res["max_ratio"] == 800.0


def test_empty_trace_handling():
    events = parse_trace_events({"traceEvents": []})
    res = find_spillover_spike(events)
    assert res["argmin_index"] == -1
    assert res["max_ratio"] == 0.0


def test_unrelated_events_filtered():
    data = {
        "traceEvents": [
            {"name": "aten::matmul", "ts": 100, "dur": 1000, "args": {}},
            {"name": "PagedAdamW8bit::step", "ts": 200, "dur": 150, "args": {"page_faults": 5, "step": 1}}
        ]
    }
    events = parse_trace_events(data)
    assert len(events) == 1
    assert events[0]["name"] == "PagedAdamW8bit::step"
