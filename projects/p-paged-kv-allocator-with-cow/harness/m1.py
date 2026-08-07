def check(workdir):
    from kv.allocator import PagedKVAllocator

    m = {"api_ok": 0.0, "allocates": 0.0, "frees": 0.0}
    try:
        a = PagedKVAllocator(10, 4)
        if a.free_count() != 10:
            return m
        m["api_ok"] = 1.0

        a.allocate_seq(1)
        for _ in range(5):
            a.append_token(1)

        bt = a.get_block_table(1)
        if len(bt) == 2 and a.free_count() == 8:
            m["allocates"] = 1.0

        a.free_seq(1)
        if a.free_count() == 10:
            m["frees"] = 1.0
    except Exception:
        pass
    return m
