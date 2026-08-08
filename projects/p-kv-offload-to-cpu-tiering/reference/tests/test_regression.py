from kvtier.cost import estimate_transfer_cost
from kvtier.policy import select_sessions_to_offload
from kvtier.tier import TierManager


def test_transfer_cost_positive():
    cost = estimate_transfer_cost(1024, 64, 10.0)
    assert cost > 0


def test_policy_eviction():
    sessions = [
        {"id": "s1", "tokens": 100, "priority": 1, "last_accessed": 10},
        {"id": "s2", "tokens": 100, "priority": 1, "last_accessed": 1},
    ]
    offloaded = select_sessions_to_offload(sessions, 100)
    assert "s2" in offloaded


def test_tier_capacity_limit():
    tm = TierManager(10)
    ok1 = tm.offload("s1", list(range(5)))
    ok2 = tm.offload("s2", list(range(10)))
    assert ok1 is True
    assert ok2 is False
