def check(workdir):
    from kv.allocator import KVAllocator
    import ref

    m = {"fragmentation_ok": 0.0}
    trace = ref.get_beam_search_trace()
    try:
        learner_fc = ref.run_trace(KVAllocator, 1000, 8, trace)
        oracle_fc = ref.run_trace(ref.OracleAllocator, 1000, 8, trace)

        if min(learner_fc) >= min(oracle_fc):
            m["fragmentation_ok"] = 1.0
    except Exception:
        pass

    return m
