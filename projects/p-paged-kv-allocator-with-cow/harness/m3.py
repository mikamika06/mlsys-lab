def check(workdir):
    from kv.allocator import PagedKVAllocator

    m = {"cow_on_append": 0.0, "cow_preserves_old": 0.0}
    try:
        a = PagedKVAllocator(10, 4)
        a.allocate_seq(1)
        for _ in range(2):
            a.append_token(1)

        a.fork_seq(1, 2)

        a.append_token(2)

        bt1 = a.get_block_table(1)
        bt2 = a.get_block_table(2)

        if bt1[0] != bt2[0] and a.free_count() == 8:
            m["cow_on_append"] = 1.0

        a.append_token(1)
        bt1_new = a.get_block_table(1)
        if bt1_new[0] == bt1[0] and a.free_count() == 8:
            m["cow_preserves_old"] = 1.0
    except Exception:
        pass
    return m
