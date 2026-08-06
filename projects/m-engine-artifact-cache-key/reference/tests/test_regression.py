import sys
sys.path.insert(0, ".")
from engcache.key import compute_cache_key
from engcache.batching import optimal_queue_delay
from engcache.coldstart import decompose_cold_start

def test_cache_key_determinism():
    cfg = {"model": "test", "precision": "fp16"}
    k1 = compute_cache_key(cfg)
    k2 = compute_cache_key(cfg)
    assert k1 == k2

def test_batching_delay_bounds():
    d = optimal_queue_delay(100.0, 500.0, 16, 0.05)
    assert d > 0.0

def test_cold_start_fractions():
    timings = {"parsing": 10.0, "optimization": 80.0, "serialization": 10.0}
    res = decompose_cold_start(timings)
    assert sum(res.values()) == 1.0
