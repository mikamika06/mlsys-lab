def identify_eviction(requests, cache_budget):
    cached = set()
    for r in requests:
        pref = set(r.get("prefix", []))
        if len(cached.union(pref)) > cache_budget:
            return r.get("id")
        cached.update(pref)
    return -1
