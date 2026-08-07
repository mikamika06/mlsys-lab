def create_cache(isolate=False):
    from prefix_cache import PrefixCache
    return PrefixCache(isolate=isolate)
