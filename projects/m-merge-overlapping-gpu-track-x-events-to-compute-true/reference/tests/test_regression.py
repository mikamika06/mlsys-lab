import sys
sys.path.insert(0, ".")
from gpuprof.events import parse_trace_events
from gpuprof.busy import compute_gpu_busy_time
from gpuprof.latency import compute_host_to_device_latencies

def test_overlapping_gpu_busy_time():
    events = [
        {"ph": "X", "cat": "gpu_op", "ts": 100.0, "dur": 50.0, "args": {"stream": 1}},
        {"ph": "X", "cat": "gpu_op", "ts": 120.0, "dur": 50.0, "args": {"stream": 2}},
    ]
    busy = compute_gpu_busy_time(events)
    assert busy == 70.0

def test_truncation_detection():
    events = [
        {"ph": "B", "pid": 1, "tid": 1, "name": "step"},
        {"ph": "X", "cat": "gpu_op", "ts": 10.0, "dur": 20.0},
    ]
    res = parse_trace_events(events)
    assert res["is_truncated"] is True
    assert res["unmatched_b_count"] == 1

def test_host_to_device_latency():
    events = [
        {"ph": "X", "cat": "host_op", "ts": 100.0, "dur": 5.0, "args": {"correlation_id": 42}},
        {"ph": "X", "cat": "gpu_op", "ts": 125.0, "dur": 30.0, "args": {"correlation_id": 42}},
    ]
    latencies = compute_host_to_device_latencies(events)
    assert latencies == {42: 25.0}
