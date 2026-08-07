import sys
sys.path.insert(0, ".")
from rollout.policy import RolloutPolicy
from rollout.manager import ModelManager


def test_policy_weight_progression():
    p = RolloutPolicy([0.2, 0.6, 1.0], 0.05)
    assert p.get_weight(0) < p.get_weight(1)
    assert p.get_weight(1) < p.get_weight(2)


def test_rollback_on_high_error():
    p = RolloutPolicy([0.2, 0.5, 1.0], 0.05)
    assert p.should_rollback(0.10) is True
    assert p.get_weight(1) == 0.0
