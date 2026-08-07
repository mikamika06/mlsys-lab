import numpy as np

def verify_cache_update(cache, update, pos):
    res = cache.copy()
    res[:, :, pos:pos+1, :] = update
    return res
