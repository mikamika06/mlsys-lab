def check(workdir):
    from kv.allocator import KVAllocator

    m = {"allocs": 0.0, "appends": 0.0, "frees": 0.0}
    try:
        a = KVAllocator(10, 4)
        s = a.allocate_sequence()
        if a.free_count() == 10 and s is not None:
            m["allocs"] = 1.0

        a.append_tokens(s, 5)
        if a.free_count() == 8 and len(a.get_block_table(s)) == 2:
            m["appends"] = 1.0

        a.free_sequence(s)
        if a.free_count() == 10:
            m["frees"] = 1.0
    except Exception:
        pass

    return m
