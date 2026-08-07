def check(workdir):
    from kv.allocator import PagedKVAllocator

    m = {"fork_shares": 0.0, "refcount_correct": 0.0}
    try:
        a = PagedKVAllocator(10, 4)
        a.allocate_seq(1)
        for _ in range(4):
            a.append_token(1)

        a.fork_seq(1, 2)
        bt1 = a.get_block_table(1)
        bt2 = a.get_block_table(2)

        if bt1 == bt2 and a.free_count() == 9:
            m["fork_shares"] = 1.0

        a.free_seq(1)
        if a.free_count() == 9:
            a.free_seq(2)
            if a.free_count() == 10:
                m["refcount_correct"] = 1.0
    except Exception:
        pass
    return m
