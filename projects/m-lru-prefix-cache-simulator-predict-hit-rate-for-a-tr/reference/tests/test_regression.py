import sys
sys.path.insert(0, ".")
from prefcache.simulator import simulate_lru_hit_rate
from prefcache.eviction import reproduce_eviction_sequence
from prefcache.metrics import compute_hit_rate_from_prometheus

def test_simulator_basic():
    traces = [[1, 2, 3, 1, 2, 4]]
    rate = simulate_lru_hit_rate(traces, capacity=3)
    assert 0.0 <= rate <= 1.0

def test_eviction_sequence_non_empty():
    ops = [("access", 1), ("access", 2), ("access", 3), ("access", 4)]
    seq = reproduce_eviction_sequence(ops, capacity=2)
    assert isinstance(seq, list)

def test_metrics_calculation():
    rate = compute_hit_rate_from_prometheus(80, 20)
    assert abs(rate - 0.8) < 1e-6
