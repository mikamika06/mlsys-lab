import ref


def check(workdir):
    from evict.tree import eviction_order
    from evict.scheduler import simulate_reclaim

    out = {"order_match": 0.0, "reclaim_match": 0.0, "safe_retention": 0.0}
    ok_order = 0
    ok_reclaim = 0
    ok_safe = 0

    for cfg in ref.CONFIGS:
        want_order = ref.eviction_order(cfg)
        got_order = eviction_order(cfg)
        if got_order == want_order:
            ok_order += 1

        want_reclaim = ref.simulate_reclaim(cfg)
        got_reclaim = simulate_reclaim(cfg)
        if got_reclaim == want_reclaim:
            ok_reclaim += 1

        try:
            assert ref.check_safety(cfg)
            ok_safe += 1
        except Exception:
            pass

    out["order_match"] = 1.0 if ok_order == len(ref.CONFIGS) else 0.0
    out["reclaim_match"] = 1.0 if ok_reclaim == len(ref.CONFIGS) else 0.0
    out["safe_retention"] = 1.0 if ok_safe == len(ref.CONFIGS) else 0.0
    return out
