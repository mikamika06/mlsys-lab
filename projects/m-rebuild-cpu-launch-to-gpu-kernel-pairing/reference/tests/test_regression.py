import sys
sys.path.insert(0, ".")
from trace_parser.flops import compute_flops

def test_compute_flops_positive():
    events = [{"name": "aten::mm", "dur": 1000, "args": {"flops": 2000000}}]
    res = compute_flops(events)
    assert "aten::mm" in res
    assert res["aten::mm"] > 0.0
