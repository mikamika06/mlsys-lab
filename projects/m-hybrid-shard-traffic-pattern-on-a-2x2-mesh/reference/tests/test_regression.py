import sys

sys.path.insert(0, ".")
from meshshard.traffic import compute_traffic
from meshshard.policy import diagnose_policy
from meshshard.strategy import select_strategy


def test_traffic_shape():
    t = compute_traffic((2, 2), {"intra_group": 2, "inter_group": 2})
    assert len(t) == 4


def test_policy_diagnosis():
    issues = diagnose_policy({"layer1": 100, "layer2": 50000}, 1000)
    assert len(issues) == 1


def test_strategy_selection():
    best = select_strategy(["FULL_SHARD", "HYBRID_SHARD"], 1000, {"FULL_SHARD": 800, "HYBRID_SHARD": 1200})
    assert best == "FULL_SHARD"
