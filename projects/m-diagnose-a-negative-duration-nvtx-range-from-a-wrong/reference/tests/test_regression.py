import sys

sys.path.insert(0, ".")
from nvtxprof.mac import analyze_mac_trace
from nvtxprof.nvtx import diagnose_nvtx_mismatches


def test_nvtx_thread_isolation():
    events = [
        {"id": 1, "thread_id": 10, "type": "push", "name": "op_a", "timestamp": 100},
        {"id": 2, "thread_id": 20, "type": "push", "name": "op_b", "timestamp": 150},
        {"id": 3, "thread_id": 10, "type": "pop", "name": None, "timestamp": 200},
    ]
    res = diagnose_nvtx_mismatches(events)
    assert len(res["ranges"]) == 1
    assert res["ranges"][0]["name"] == "op_a"
    assert res["ranges"][0]["thread_id"] == 10
    assert len(res["unclosed_pushes"]) == 1
    assert res["unclosed_pushes"][0]["thread_id"] == 20


def test_mac_self_time_subtraction():
    events = [
        {"name": "outer", "ph": "X", "ts": 0.0, "dur": 100.0, "tid": 1},
        {"name": "inner", "ph": "X", "ts": 10.0, "dur": 40.0, "tid": 1},
    ]
    phases = ["outer", "inner"]
    res = analyze_mac_trace(events, phases)
    metrics = res["phase_metrics"]
    assert metrics["outer"]["self_time"] == 60.0
    assert metrics["inner"]["self_time"] == 40.0
    assert metrics["outer"]["total_time"] == 100.0
