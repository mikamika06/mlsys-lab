def check(workdir):
    from kvtier.tier import TieredStorage
    m = {"capacity_and_latency_ok": 0.0}
    ts = TieredStorage(1, 2)
    ts.access("s1")
    ts.evict_to_cpu("s1")
    ts.access("s2")
    if ts.bring_to_gpu("s1"):
        m["capacity_and_latency_ok"] = 1.0
    return m
