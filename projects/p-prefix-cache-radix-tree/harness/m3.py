def check(workdir):
    try:
        from prefix_cache.cache import PrefixCache
    except ImportError:
        return {"lru_ok": 0.0, "ref_count_respected": 0.0}

    c = PrefixCache(16)
    m = {"lru_ok": 0.0, "ref_count_respected": 0.0}
    try:
        b1, b2, b3 = (1,), (2,), (3,)
        c.insert("t1", [b1], [10])
        c.insert("t1", [b1, b2], [10, 20])
        c.insert("t1", [b3], [30])

        c.inc_ref([30])

        e1 = c.evict()
        if e1 == 20:
            e2 = c.evict()
            if e2 == 10:
                m["lru_ok"] = 1.0

        e3 = c.evict()
        if e3 is None:
            c.dec_ref([30])
            if c.evict() == 30:
                m["ref_count_respected"] = 1.0

    except Exception:
        pass
    return m
