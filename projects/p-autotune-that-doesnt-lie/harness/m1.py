def check(workdir):
    from autotune.cache import make_cache_key
    m = {"cache_miss_detected": 0.0}
    k1 = make_cache_key((128, 128), (128, 1))
    k2 = make_cache_key((128, 64), (64, 1))
    if k1 != k2:
        m["cache_miss_detected"] = 1.0
    return m
