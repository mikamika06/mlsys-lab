import sys
sys.path.insert(0, ".")
from mesh.traffic import simulate_traffic
from mesh.policy import diagnose_wrap_policy
from mesh.strategy import select_strategy


def test_traffic_pattern_invariant():
    res = simulate_traffic("HYBRID_SHARD", (2, 2))
    assert res["total_traffic"] > 0
    assert res["max_link_load"] <= res["total_traffic"]


def test_policy_diagnosis_accuracy():
    res = diagnose_wrap_policy([{"name": "l1", "size": 2000, "wrapped": True}], {"min_size": 1024})
    assert res["is_valid"] is True


def test_strategy_memory_budget_bound():
    strat = select_strategy(10000, 5000, (2, 2))
    assert strat in ["HYBRID_SHARD", "FULL_SHARD", "NO_SHARD_OOM"]
