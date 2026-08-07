def check(workdir):
    from kvtier.prefetch import Prefetcher
    m = {"prefetch_ok": 0.0}
    pf = Prefetcher()
    if pf.should_prefetch("s1", [1.0, 2.0]) is not None:
        m["prefetch_ok"] = 1.0
    return m
