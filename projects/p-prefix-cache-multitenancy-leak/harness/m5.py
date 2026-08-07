def check(workdir):
    from cache import PrefixCache
    import ref

    m = {"zero_intersections": 0.0}
    alloc = ref.BlockAllocator()
    c = PrefixCache(4, alloc, isolation=True, shared_system=True)

    sys_toks = [1, 2, 3, 4, 5, 6, 7, 8]
    c.insert(sys_toks, "A", is_system=True)

    a_usr = [10, 11, 12, 13]
    c.insert(sys_toks + a_usr, "A", is_system=False)

    b_usr = [10, 11, 12, 13]
    b_matched = c.match(sys_toks + b_usr, "B")

    if len(b_matched) == 2:
        m["zero_intersections"] = 1.0

    return m
