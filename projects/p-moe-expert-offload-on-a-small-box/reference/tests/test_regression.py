import sys
sys.path.insert(0, ".")
from moe.cache import ExpertCache
from moe.policy import evaluate_latency

def test_cache_capacity_respected():
    c = ExpertCache(2500)
    c.access(1, 1000)
    c.access(2, 1000)
    c.access(3, 1000)
    assert 1 not in c.cache
    assert 2 in c.cache
    assert 3 in c.cache

def test_latency_evaluation():
    trace = [{"activated": [1, 2], "size": 1000}]
    c = ExpertCache(5000)
    lat = evaluate_latency(trace, c, 5000)
    assert lat > 0
