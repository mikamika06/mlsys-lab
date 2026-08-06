"""Regression tests for profiling utilities."""
from prof.cache import compute_cache_hit_ratio
from prof.analysis import compare_execution

def test_cache_hit_ratio_basic():
    seq = [1, 2, 1, 2]
    ratio = compute_cache_hit_ratio(seq, cache_capacity=2)
    assert ratio == 0.5

def test_compare_execution_basic():
    eager = {"ops": 10, "size": 100}
    compiled = {"ops": 5, "size": 50}
    res = compare_execution(eager, compiled)
    assert res["size_ratio"] == 0.5
    assert res["op_ratio"] == 0.5
