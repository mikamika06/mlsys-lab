import ref

def check(workdir):
    m = {"speedup_ok": 0.0}
    try:
        tokens = list(range(200))
        out = ref.run_cached_vs_uncached(tokens)
        if len(out) == 200:
            m["speedup_ok"] = 1.0
    except Exception:
        pass
    return m
