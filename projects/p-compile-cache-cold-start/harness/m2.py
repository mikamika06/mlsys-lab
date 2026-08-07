import ref

def check(workdir):
    from compcache.cache import CompilationCache
    c = CompilationCache()
    c.store("k1", b"compiled_blob")
    hits = 0
    total = 10
    for i in range(total):
        if c.lookup("k1") is not None:
            hits += 1
    return ref.get_oracle_cache(hits, total)
