def check(workdir):
    from kv.allocator import KVAllocator

    m = {"fork_ok": 0.0, "refcount_ok": 0.0}
    try:
        a = KVAllocator(10, 4)
        s1 = a.allocate_sequence()
        a.append_tokens(s1, 6)
        s2 = a.fork_sequence(s1)

        t1 = a.get_block_table(s1)
        t2 = a.get_block_table(s2)
        if t1 == t2 and a.free_count() == 8:
            m["fork_ok"] = 1.0

        b0 = t1[0]
        if a.get_block_refcount(b0) == 2:
            m["refcount_ok"] = 1.0
    except Exception:
        pass

    return m
