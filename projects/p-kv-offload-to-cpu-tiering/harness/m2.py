def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from kvtier import policy
    import ref

    m = {"eviction_order_ok": 0.0, "retains_hot": 0.0}

    sessions = [
        {"id": "old", "tokens": 50, "priority": 1, "last_accessed": 1},
        {"id": "hot", "tokens": 50, "priority": 10, "last_accessed": 100},
    ]

    res = policy.select_sessions_to_offload(sessions, 50)
    expected = ref.oracle_select_offload(sessions, 50)

    if res == expected and "old" in res and "hot" not in res:
        m["eviction_order_ok"] = 1.0
        m["retains_hot"] = 1.0

    return m
