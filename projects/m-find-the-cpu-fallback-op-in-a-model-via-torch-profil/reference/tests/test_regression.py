import sys
sys.path.insert(0, ".")
from mps_diag.profiler import find_fallback_ops
from mps_diag.latency import measure_latency_cliff


def test_find_fallback_ops_detects_nonzero():
    events = [
        {"name": "aten::nonzero", "dur": 1000, "cat": "cpu_op"},
        {"name": "aten::to", "dur": 500, "cat": "Memcpy"}
    ]
    ops = find_fallback_ops(events)
    assert len(ops) > 0, "failed to detect fallback operations"
    assert "aten::nonzero" in ops


def test_latency_cliff_ratio():
    ratio = measure_latency_cliff(200.0, 10.0)
    assert ratio == 20.0, f"expected ratio 20.0, got {ratio}"
