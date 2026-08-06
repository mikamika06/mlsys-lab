import sys
sys.path.insert(0, ".")
from tcache.hashing import compute_cache_key
from tcache.metrics import compute_metrics
from tcache.safety import is_safe_config

def test_cache_key_deterministic():
    r1 = {"a": 1, "b": 2}
    r2 = {"b": 2, "a": 1}
    assert compute_cache_key(r1) == compute_cache_key(r2)

def test_metrics_non_negative():
    counters = {"hits": 10, "misses": 5, "hit_latency_ms": 2.0, "miss_latency_ms": 10.0, "bytes_per_entry": 512}
    res = compute_metrics(counters)
    assert res["saved_latency_ms"] >= 0
    assert res["memory_cost_bytes"] >= 0

def test_safety_flagging():
    assert is_safe_config({"stochastic": False, "dynamic_state": False}) is True
    assert is_safe_config({"stochastic": True, "dynamic_state": False}) is False
    assert is_safe_config({"stochastic": False, "dynamic_state": True}) is False
