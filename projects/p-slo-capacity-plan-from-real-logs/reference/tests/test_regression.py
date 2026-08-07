import sys
import pytest

sys.path.insert(0, ".")
from capacity.planner import (
    compute_required_replicas,
    compute_cost_per_million_tokens,
    compute_prefix_cache_impact,
)


def test_required_replicas_headroom():
    reps = compute_required_replicas(target_rps=40, single_replica_capacity=10, headroom_factor=1.2)
    assert reps == 5


def test_cost_calculation():
    cost = compute_cost_per_million_tokens(
        replica_count=4,
        hourly_cost_per_replica=2.0,
        rps=40,
        avg_output_tokens=100,
    )
    assert round(cost, 4) == 0.5556


def test_prefix_cache_scaling():
    res = compute_prefix_cache_impact(
        target_rps=40,
        base_single_replica_capacity=10,
        hit_rate=0.5,
        speedup_factor=2.0,
        headroom_factor=1.2,
    )
    assert res["effective_single_capacity"] == 15.0
    assert res["required_replicas"] == 4
