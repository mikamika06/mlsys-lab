def test_no_zero_refcounts():
    from kv.allocator import KVAllocator
    a = KVAllocator(10, 4)
    s1 = a.allocate_sequence()
    a.append_tokens(s1, 5)
    s2 = a.fork_sequence(s1)
    a.free_sequence(s1)
    for b in a.get_block_table(s2):
        assert a.get_block_refcount(b) > 0
