import sys

sys.path.insert(0, ".")
from condcheck.decision import decide_branch_strategy
from condcheck.measurement import compute_latency_ratio


def test_ratio_calculation():
    r = compute_latency_ratio(10, 64)
    assert isinstance(r, float)
    assert r > 0.0


def test_decision_output_validity():
    case = {"id": 1, "ops": 5, "tensor_elements": 128}
    res = decide_branch_strategy(case)
    assert res in ("cond", "where")


def test_heavy_ops_prefers_cond():
    case = {"id": 2, "ops": 5000, "tensor_elements": 128}
    assert decide_branch_strategy(case) == "cond"
