def check(workdir):
    from kv.allocator import KVAllocator
    import ref

    m = {"no_leaks": 0.0, "matches_oracle": 0.0}
    trace = ref.get_random_trace(10000)
    try:
        learner_fc = ref.run_trace(KVAllocator, 200, 16, trace)
        oracle_fc = ref.run_trace(ref.OracleAllocator, 200, 16, trace)

        if learner_fc[-1] == 200:
            m["no_leaks"] = 1.0
        if learner_fc == oracle_fc:
            m["matches_oracle"] = 1.0
    except Exception:
        pass

    return m
