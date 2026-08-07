def check(workdir):
    from moe.cache import ExpertCache

    m = {"cache_hit_rate": 0.0}
    cache = ExpertCache(2000)
    hit1 = cache.access(1, 1000)
    hit2 = cache.access(1, 1000)
    if not hit1 and hit2:
        m["cache_hit_rate"] = 1.0
    return m
