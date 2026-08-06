def check(workdir):
    out = {"latency_ratio": 1.0}
    try:
        from flexmask.caching import BlockMaskCache
    except Exception as e:
        out["_note"] = f"import error: {e}"
        return out

    cache = BlockMaskCache()
    calls = 0
    def factory(s, b):
        nonlocal calls
        calls += 1
        return {"seq": s, "block": b}

    steps = 50
    for _ in range(steps):
        cache.get_or_create(64, 16, factory)

    ratio = float(calls) / float(steps)
    out["latency_ratio"] = ratio
    if calls > 1:
        out["_note"] = f"cache failed to cache, factory called {calls} times"
    return out
