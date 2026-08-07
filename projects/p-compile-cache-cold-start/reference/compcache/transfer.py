import pickle

def serialize_cache(cache):
    """Serialize compilation cache into bytes."""
    return pickle.dumps(cache.store_map)

def deserialize_cache(data):
    """Deserialize compilation cache from bytes."""
    import pickle
    from compcache.cache import CompilationCache
    c = CompilationCache()
    c.store_map = pickle.loads(data)
    return c
