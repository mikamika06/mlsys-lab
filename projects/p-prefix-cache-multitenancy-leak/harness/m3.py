def check(workdir):
    from cache import PrefixCache
    import ref

    m = {"hr_drops": 0.0}

    alloc1 = ref.BlockAllocator()
    c1 = PrefixCache(4, alloc1, isolation=False)

    alloc2 = ref.BlockAllocator()
    c2 = PrefixCache(4, alloc2, isolation=True)

    trace = ref.get_trace()

    h1, t1 = ref.run_trace(c1, trace)
    h2, t2 = ref.run_trace(c2, trace)

    if t1 > 0 and t2 > 0 and (h2 / t2) < (h1 / t1):
        m["hr_drops"] = 1.0

    return m
