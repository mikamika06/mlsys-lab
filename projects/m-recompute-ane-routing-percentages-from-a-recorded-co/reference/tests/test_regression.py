import sys
sys.path.insert(0, ".")
from aneplan.latency import compare_latencies


def test_latency_comparison_non_negative():
    plan = {"ops": [{"op_name": "conv1", "estimated_cost": 1.5}]}
    measured = {"conv1": 1.4}
    res = compare_latencies(plan, measured)
    assert res["conv1"]["rel_diff"] >= 0.0


def test_latency_comparison_keys():
    plan = {"ops": [{"op_name": "matmul1", "estimated_cost": 2.0}]}
    measured = {"matmul1": 2.1}
    res = compare_latencies(plan, measured)
    assert "matmul1" in res
