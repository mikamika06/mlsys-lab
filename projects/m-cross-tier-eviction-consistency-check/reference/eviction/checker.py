"""Cross-tier cache state consistency checker implementation."""

def check_cross_tier_consistency(t0_state, t1_state):
    violations = []
    t0_ids = set(t0_state.keys())
    t1_ids = set(t1_state.keys())

    for block_id in t0_ids:
        t0_meta = t0_state[block_id]
        if block_id in t1_ids:
            t1_meta = t1_state[block_id]
            if t0_meta.get("hash") != t1_meta.get("hash"):
                violations.append((block_id, "HASH_MISMATCH"))
            if t0_meta.get("dirty", False) and not t1_meta.get("stale", False):
                violations.append((block_id, "DIRTY_T0_CLEAN_T1"))

    for block_id in t1_ids:
        t1_meta = t1_state[block_id]
        if t1_meta.get("status") == "EVICTED_DIRTY":
            violations.append((block_id, "UNSYNCED_EVICTION"))

    return len(violations) == 0, violations
