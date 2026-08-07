def check(workdir):
    from kv.allocator import KVAllocator

    m = {"cow_triggers": 0.0, "cow_preserves_old": 0.0}
    try:
        a = KVAllocator(10, 4)
        s1 = a.allocate_sequence()
        a.append_tokens(s1, 2)
        b_orig = a.get_block_table(s1)[0]
        s2 = a.fork_sequence(s1)

        a.append_tokens(s1, 1)
        t1 = a.get_block_table(s1)
        t2 = a.get_block_table(s2)

        if t1[0] != t2[0] and t2[0] == b_orig:
            m["cow_preserves_old"] = 1.0

        if a.get_block_refcount(t1[0]) == 1 and a.get_block_refcount(t2[0]) == 1:
            if a.free_count() == 8:
                m["cow_triggers"] = 1.0
    except Exception:
        pass

    return m
