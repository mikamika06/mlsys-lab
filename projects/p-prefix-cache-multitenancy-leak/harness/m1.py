def check(workdir):
    from cache import PrefixCache
    import ref

    m = {"leak_present": 0.0}
    alloc = ref.BlockAllocator()
    c = PrefixCache(4, alloc, isolation=False)

    c.insert([1, 2, 3, 4], "A")
    if len(c.match([1, 2, 3, 4], "B")) == 1:
        m["leak_present"] = 1.0

    return m
