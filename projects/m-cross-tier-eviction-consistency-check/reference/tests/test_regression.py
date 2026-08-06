"""Regression tests for cross-tier eviction consistency."""

from eviction.manager import CrossTierEvictionManager
from eviction.checker import check_cross_tier_consistency

def test_eviction_consistency():
    mgr = CrossTierEvictionManager(t0_capacity=2, t1_capacity=2)
    mgr.register_block("b1", "hash_a", tier=0)
    mgr.evict_from_t0("b1", preserve_in_t1=False)
    t0_state, t1_state = mgr.get_tier_states()
    assert "b1" not in t0_state
    assert "b1" not in t1_state
    valid, violations = check_cross_tier_consistency(t0_state, t1_state)
    assert valid
    assert len(violations) == 0
