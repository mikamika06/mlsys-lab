import sys
sys.path.insert(0, ".")
from trace_parser.proton import analyze_proton

def test_exclusive_time_sum():
    events = [
        {"time": 0.0, "type": "enter", "region": "outer"},
        {"time": 10.0, "type": "enter", "region": "inner"},
        {"time": 20.0, "type": "exit", "region": "inner"},
        {"time": 30.0, "type": "exit", "region": "outer"}
    ]
    res = analyze_proton(events)
    total_pct = sum(res.values())
    assert abs(total_pct - 100.0) < 1e-5, f"Expected 100%, got {total_pct}%"
