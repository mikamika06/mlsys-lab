import sys
sys.path.insert(0, ".")
from autocast.engine import simulate_stack
from autocast.analyzer import analyze_regions
from autocast.policy import resolve_precision


def test_simulate_stack_basic():
    events = [
        {"type": "push", "device": "cuda", "dtype": "float16", "enabled": True},
        {"type": "push", "device": "cuda", "dtype": "bfloat16", "enabled": False},
        {"type": "pop"}
    ]
    res = simulate_stack(events)
    assert len(res) == 3
    assert res[1]["enabled"] is False
    assert res[2]["enabled"] is True


def test_policy_resolution():
    stack = [{"enabled": False, "dtype": "float16"}]
    assert resolve_precision(stack, "float16") == "float32"


def test_analyzer_detection():
    trace = [
        {"event": "push", "enabled": True, "dtype": "float16"},
        {"event": "op", "sensitive": True, "used_dtype": "float16"}
    ]
    res = analyze_regions(trace)
    assert res["violations"] == 1
