import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    import trtep.audit as audit
    import trtep.cache as cache

    m = {"cache_miss_cold": 0.0, "cache_hit_warm": 0.0, "hash_consistency": 0.0}
    try:
        c = cache.EngineCache()
        g = ref.build_benchmark_graph()
        subs = audit.partition_graph(g, ref.DEFAULT_SUPPORTED_OPS)
        sub0 = subs[0]
        sub1 = subs[1]
    except Exception:
        return m

    try:
        h0 = c.compute_hash(sub0)
        h0_again = c.compute_hash(sub0)
        h1 = c.compute_hash(sub1)
        if isinstance(h0, str) and h0 == h0_again and h0 != h1:
            m["hash_consistency"] = 1.0
    except Exception:
        return m

    build_calls = 0

    def mock_builder(sub):
        nonlocal build_calls
        build_calls += 1
        return {"engine_blob": f"compiled_{sub.sub_id}"}

    try:
        res1, is_hit1 = c.build_or_load(sub0, mock_builder)
        if not is_hit1 and build_calls == 1 and res1["engine_blob"] == "compiled_0":
            m["cache_miss_cold"] = 1.0

        res2, is_hit2 = c.build_or_load(sub0, mock_builder)
        if is_hit2 and build_calls == 1 and res2["engine_blob"] == "compiled_0":
            m["cache_hit_warm"] = 1.0
    except Exception:
        pass

    return m
