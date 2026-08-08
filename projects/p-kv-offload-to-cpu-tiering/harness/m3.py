def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from kvtier import prefetch, tier
    import ref

    m = {"prefetch_triggered": 0.0, "latency_reduced": 0.0}

    tm = tier.TierManager(100)
    tm.offload("s1", [1, 2, 3])

    queue = [{"session_id": "s1"}]
    res = prefetch.should_prefetch("s1", queue, tm)
    expected = ref.oracle_should_prefetch("s1", queue, tm)

    if res == expected and res is True:
        m["prefetch_triggered"] = 1.0
        m["latency_reduced"] = 1.0

    return m
