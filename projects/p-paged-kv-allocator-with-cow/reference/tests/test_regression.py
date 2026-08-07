def test_no_zero_refcounts():
    from kv.allocator import PagedKVAllocator
    a = PagedKVAllocator(10, 4)
    a.allocate_seq(1)
    a.append_token(1)

    a.fork_seq(1, 2)
    a.free_seq(1)

    table = a.get_block_table(2)
    for b in table:
        assert a.ref_count[b] > 0, "Block has zero ref count"
