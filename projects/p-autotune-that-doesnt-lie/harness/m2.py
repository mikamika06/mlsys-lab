def check(workdir):
    from autotune.cache import make_cache_key
    m = {"keys_unique": 0.0}
    shapes = [(128, 128), (128, 64), (64, 64)]
    strides = [(128, 1), (64, 1), (64, 1)]
    keys = {make_cache_key(s, st) for s, st in zip(shapes, strides)}
    if len(keys) == len(shapes):
        m["keys_unique"] = 1.0
    return m
