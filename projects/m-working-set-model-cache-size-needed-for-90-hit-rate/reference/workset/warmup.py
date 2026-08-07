def warmup_curve(trace, cache_size):
    cache = []
    hits = []
    for item in trace:
        if item in cache:
            hits.append(1)
            cache.remove(item)
            cache.append(item)
        else:
            hits.append(0)
            if len(cache) >= cache_size:
                cache.pop(0)
            cache.append(item)
    return hits
