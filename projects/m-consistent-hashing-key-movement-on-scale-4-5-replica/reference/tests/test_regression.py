from chash.router import ConsistentHashRing, calculate_remapping_fraction
from chash.affinity import SessionAffinityRouter


def test_routing_invariants():
    ring4 = ConsistentHashRing(["r1", "r2", "r3", "r4"], num_tokens=100)
    ring5 = ConsistentHashRing(["r1", "r2", "r3", "r4", "r5"], num_tokens=100)

    sample_keys = [f"prompt_key_{i}" for i in range(1000)]
    remap_frac = calculate_remapping_fraction(ring4, ring5, sample_keys)

    assert 0.10 <= remap_frac <= 0.35, f"Remapping fraction unexpected: {remap_frac}"

    router = SessionAffinityRouter(ring4, ttl_seconds=10)
    rep1 = router.route("session_a", "key_1", current_time=0)
    rep2 = router.route("session_a", "key_2", current_time=5)
    assert rep1 == rep2, "Affinity failed within TTL window"

    rep3 = router.route("session_a", "key_2", current_time=20)
    assert rep3 == ring4.get_replica("key_2"), "Expired affinity did not fallback to hash lookup"
