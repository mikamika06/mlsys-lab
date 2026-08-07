def check(workdir):
    try:
        from prefix_cache.cache import PrefixCache
    except ImportError:
        return {"insert_ok": 0.0, "match_longest": 0.0}

    c = PrefixCache(16)
    m = {"insert_ok": 0.0, "match_longest": 0.0}
    try:
        b1, b2, b3 = (1,), (2,), (3,)
        c.insert("t1", [b1, b2], [10, 20])

        r1 = c.match("t1", [b1, b2])
        if r1 == [10, 20]:
            m["insert_ok"] = 1.0

        r2 = c.match("t1", [b1, b2, b3])
        if r2 == [10, 20]:
            m["match_longest"] = 1.0

        r3 = c.match("t1", [b1, b3])
        if r3 != [10]:
            m["match_longest"] = 0.0

    except Exception:
        pass
    return m
