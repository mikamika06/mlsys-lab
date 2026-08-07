def check(workdir):
    try:
        from prefix_cache.cache import PrefixCache
    except ImportError:
        return {"chain_ok": 0.0, "tenant_hashed": 0.0}

    import ref
    ref.dummy()

    c = PrefixCache(16)
    try:
        bA = (1, 2)
        bB = (3, 4)
        bC = (5, 6)

        h1 = c.compute_hashes("t1", [bA, bB])
        h2 = c.compute_hashes("t1", [bA])
        h3 = c.compute_hashes("t2", [bA, bB])
        h4 = c.compute_hashes("t1", [bC, bB])

        m = {"chain_ok": 0.0, "tenant_hashed": 0.0}

        if len(h1) == 2 and len(h2) == 1 and h1[0] == h2[0]:
            if h1[1] != h4[1]:
                m["chain_ok"] = 1.0

        if len(h1) > 0 and len(h3) > 0 and h1[0] != h3[0]:
            m["tenant_hashed"] = 1.0

        return m
    except Exception:
        return {"chain_ok": 0.0, "tenant_hashed": 0.0}
