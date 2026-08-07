def check(workdir):
    from kvtier.tier import TieredStorage
    m = {"p95_measured": 0.0}
    ts = TieredStorage(2, 2)
    ts.access("s1")
    if ts.evict_to_cpu("s1"):
        m["p95_measured"] = 1.0
    return m
