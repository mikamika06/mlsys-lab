import sys

sys.path.insert(0, ".")
from profiler.taxonomy import classify_mechanisms, calculate_miss_bound, rank_profilers


def test_classification_keys():
    tools = [("tool_a", "sampling"), ("tool_b", "instrumentation")]
    res = classify_mechanisms(tools)
    assert "tool_a" in res
    assert res["tool_a"] == "sampling"


def test_miss_bound_bounds():
    val = calculate_miss_bound(10.0, 2.0, 1000.0)
    assert 0.0 <= val <= 1.0


def test_rank_profilers_order():
    profilers = ["a", "b"]
    metrics = {"a": 10.0, "b": 20.0}
    res = rank_profilers(profilers, metrics)
    assert res == ["a", "b"]
