def sweep_cache_limit(shapes, cache_size_limits):
    results = []
    for limit in cache_size_limits:
        seen = set()
        recompiles = 0
        fallbacks = 0
        for s in shapes:
            if s in seen:
                continue
            if len(seen) < limit:
                recompiles += 1
                seen.add(s)
            else:
                fallbacks += 1
        results.append({
            "cache_size_limit": limit,
            "recompiles": recompiles,
            "eager_fallbacks": fallbacks,
            "cache_exhausted": len(seen) >= limit and fallbacks > 0
        })
    return results
