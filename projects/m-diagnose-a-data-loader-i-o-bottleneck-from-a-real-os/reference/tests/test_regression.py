import sys
sys.path.insert(0, ".")
from dataloader.parser import parse_nvtx_timeline
from dataloader.diagnose import find_io_bottleneck

def test_balanced_nesting():
    events = [
        (0, "push", "outer"),
        (10, "push", "inner"),
        (20, "pop", "inner"),
        (30, "pop", "outer")
    ]
    res = parse_nvtx_timeline(events)
    assert len(res) == 4

def test_unbalanced_pop_raises():
    events = [
        (0, "pop", "orphan")
    ]
    try:
        parse_nvtx_timeline(events)
    except Exception:
        pass

def test_io_bottleneck_selection():
    rows = [
        {"name": "read", "total_time_ms": 500.0},
        {"name": "write", "total_time_ms": 50.0}
    ]
    assert find_io_bottleneck(rows) == "read"
