"""Milestone 2 harness check."""

def check(workdir):
    from eviction.manager import CrossTierEvictionManager
    from eviction.checker import check_cross_tier_consistency

    out = {"evictions_correct": 0.0, "invariants_preserved": 0.0}

    try:
        mgr = CrossTierEvictionManager(t0_capacity=3, t1_capacity=3)
        for i in range(3):
            mgr.register_block(f"b{i}", f"hash_{i}", tier=0)

        # Evict b0 with preserve_in_t1=True
        mgr.evict_from_t0("b0", preserve_in_t1=True)
        t0_s, t1_s = mgr.get_tier_states()
        if "b0" not in t0_s and "b0" in t1_s:
            # Evict b1 with preserve_in_t1=False
            mgr.evict_from_t0("b1", preserve_in_t1=False)
            t0_s, t1_s = mgr.get_tier_states()
            if "b1" not in t0_s and "b1" not in t1_s:
                out["evictions_correct"] = 1.0

        valid, violations = check_cross_tier_consistency(t0_s, t1_s)
        if valid and len(violations) == 0:
            out["invariants_preserved"] = 1.0
        else:
            out["_note"] = f"Invariants broken: {violations}"
    except Exception as e:
        out["_note"] = f"Execution error: {type(e).__name__}: {str(e)}"

    return out
