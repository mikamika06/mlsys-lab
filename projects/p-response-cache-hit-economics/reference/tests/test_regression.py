"""Learner regression test suite verifying economics and cache bounds."""

import sys

sys.path.insert(0, ".")

from cache.analyzer import analyze_trace, calculate_memory_footprint
from cache.economics import compute_breakeven_hit_rate, compute_net_savings, evaluate_cache_viability
from cache.eviction import CacheSimulator


def test_zero_capacity_handling():
    sim = CacheSimulator(capacity=0, policy="lru")
    assert not sim.access("k1")
    assert not sim.access("k1")
    stats = sim.get_stats()
    assert stats["hits"] == 0
    assert stats["size"] == 0


def test_breakeven_bounds():
    be = compute_breakeven_hit_rate(compute_cost_per_req=0.1, total_requests=100, total_memory_cost=5.0)
    assert 0.0 <= be <= 1.0
    assert abs(be - 0.5) < 1e-6


def test_eviction_lru_behavior():
    sim = CacheSimulator(capacity=2, policy="lru")
    sim.access("a")
    sim.access("b")
    sim.access("a")
    sim.access("c")
    assert "b" not in sim.cache
    assert "a" in sim.cache
    assert "c" in sim.cache


def test_viability_decision():
    trace = [{"key": f"k{i % 5}"} for i in range(100)]
    res = evaluate_cache_viability(capacity=10, trace=trace, compute_cost_per_req=1.0, memory_cost_per_entry=2.0)
    assert res["should_enable"] is True
