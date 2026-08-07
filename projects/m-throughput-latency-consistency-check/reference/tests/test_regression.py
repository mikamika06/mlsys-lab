import sys
sys.path.insert(0, ".")
from perf.analyzer import validate_trace

def test_valid_trace_passes():
    events = []
    for i in range(10):
        events.append({
            "host_start": i * 0.1,
            "host_end": i * 0.1 + 0.01,
            "device_start": i * 0.1 + 0.02,
            "device_end": i * 0.1 + 0.1
        })
    validate_trace(events, concurrency=1, tol=0.05)

def test_inconsistent_trace_fails():
    events = []
    for i in range(10):
        events.append({
            "host_start": i * 0.1,
            "host_end": i * 0.1 + 0.01,
            "device_start": i * 0.1 + 0.02,
            "device_end": i * 0.1 + 1.0
        })
    try:
        validate_trace(events, concurrency=1, tol=0.05)
        assert False
    except ValueError:
        pass
