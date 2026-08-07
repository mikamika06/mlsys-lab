import ref

def check(workdir):
    from compcache.cache import CompilationCache
    from compcache.transfer import serialize_cache, deserialize_cache
    c = CompilationCache()
    c.store("k1", b"blob")
    data = serialize_cache(c)
    c2 = deserialize_cache(data)
    val = c2.lookup("k1")
    oracle = ref.get_oracle_transfer(data)
    if val != b"blob":
        oracle["transfer_ok"] = 0.0
    return oracle
