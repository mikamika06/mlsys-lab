def check(workdir):
    from cache import PrefixCache
    import ref

    m = {"system_shared": 0.0, "user_isolated": 0.0}
    alloc = ref.BlockAllocator()
    c = PrefixCache(4, alloc, isolation=True, shared_system=True)

    sys_tokens = [1, 2, 3, 4]
    c.insert(sys_tokens, "A", is_system=True)

    if len(c.match(sys_tokens, "B")) == 1:
        m["system_shared"] = 1.0

    usr_tokens = [5, 6, 7, 8]
    c.insert(sys_tokens + usr_tokens, "A", is_system=False)

    matched_b = c.match(sys_tokens + usr_tokens, "B")
    if len(matched_b) == 1:
        m["user_isolated"] = 1.0

    return m
