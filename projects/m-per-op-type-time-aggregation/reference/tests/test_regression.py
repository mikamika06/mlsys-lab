import sys
sys.path.insert(0, ".")
from ortperf.agg import aggregate_op_types
from ortperf.memcpy import locate_boundary_memcpys
from ortperf.overhead import compute_overhead

PROFILE = {
    "nodes": [
        {"name": "N0", "op_type": "MatMul", "dur": 100.0, "ep": "CPU"},
        {"name": "N1", "op_type": "MemcpyFromHost", "dur": 10.0, "ep": "CUDA"},
        {"name": "N2", "op_type": "Add", "dur": 5.0, "ep": "CUDA"}
    ],
    "session_duration": 120.0
}

def test_aggregation_sums_correctly():
    res = aggregate_op_types(PROFILE)
    assert res["MatMul"] == 100.0
    assert res["MemcpyFromHost"] == 10.0
    assert res["Add"] == 5.0

def test_boundary_memcpys_found():
    res = locate_boundary_memcpys(PROFILE)
    assert len(res) == 1
    assert res[0]["op_type"] == "MemcpyFromHost"

def test_overhead_non_negative():
    oh = compute_overhead(PROFILE)
    assert oh >= 0.0
