import sys

sys.path.insert(0, ".")

from mlplan.analysis import find_ane_rejections
from mlplan.plan import parse_compute_plan
from mlplan.profile import routing_fractions

SAMPLE_PLAN = {
    "operations": [
        {"id": "conv1", "type": "conv2d", "dispatched_device": "ANE", "estimated_cost": 5.0, "ane_supported": True, "ane_rejection_reason": None},
        {"id": "norm1", "type": "layernorm", "dispatched_device": "GPU", "estimated_cost": 2.0, "ane_supported": False, "ane_rejection_reason": "UNSUPPORTED_LAYER"},
        {"id": "custom1", "type": "custom_op", "dispatched_device": "CPU", "estimated_cost": 10.0, "ane_supported": False, "ane_rejection_reason": "DYNAMIC_SHAPE"}
    ]
}


def test_routing_fractions_sum_to_one():
    ops = parse_compute_plan(SAMPLE_PLAN)
    fracs = routing_fractions(ops)
    total = sum(fracs.values())
    assert abs(total - 1.0) < 1e-6
    assert abs(fracs["ANE"] - 1.0 / 3.0) < 1e-5
    assert abs(fracs["GPU"] - 1.0 / 3.0) < 1e-5
    assert abs(fracs["CPU"] - 1.0 / 3.0) < 1e-5


def test_find_ane_rejections_detects_all_rejections():
    ops = parse_compute_plan(SAMPLE_PLAN)
    rejections = find_ane_rejections(ops)
    assert len(rejections) == 2
    reasons = [r["reason"] for r in rejections]
    assert "UNSUPPORTED_LAYER" in reasons
    assert "DYNAMIC_SHAPE" in reasons


def test_rejections_contain_cost_and_device():
    ops = parse_compute_plan(SAMPLE_PLAN)
    rejections = find_ane_rejections(ops)
    for r in rejections:
        assert "cost" in r
        assert "device" in r
        assert r["device"] in ("GPU", "CPU")
