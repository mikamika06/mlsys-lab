def check(workdir):
    from cache import PrefixCache
    import ref

    m = {"leak_fixed": 0.0, "self_match": 0.0}
    alloc = ref.BlockAllocator()
    c = PrefixCache(4, alloc, isolation=True)

    c.insert([1, 2, 3, 4], "A")

    if len(c.match([1, 2, 3, 4], "B")) == 0:
        m["leak_fixed"] = 1.0

    if len(c.match([1, 2, 3, 4], "A")) == 1:
        m["self_match"] = 1.0

    return m
