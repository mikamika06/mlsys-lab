def check(workdir):
    try:
        from prefix_cache.cache import PrefixCache
    except ImportError:
        return {"isolation_ok": 0.0}

    c = PrefixCache(16)
    m = {"isolation_ok": 0.0}
    try:
        b1, b2 = (1,), (2,)
        c.insert("t1", [b1, b2], [10, 20])

        r1 = c.match("t2", [b1, b2])
        if r1 == []:
            m["isolation_ok"] = 1.0
    except Exception:
        pass
    return m
