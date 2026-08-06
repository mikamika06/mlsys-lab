import ref


def check(workdir):
    from coldcache.protocol import ColdCacheProtocol

    out = {"cold_state_flushed": 0.0}
    proto = ColdCacheProtocol(memory_size=512)

    proto.execute_request([10, 20, 30])
    gen1 = proto.reset_gpu_allocator()
    proto.invalidate_host_cache()

    res = proto.execute_request([10, 20, 30])

    if not res["hit"] and res["generation"] == gen1 and gen1 > 0:
        out["cold_state_flushed"] = 1.0
    else:
        out["_note"] = f"Request hit warm cache or invalid gen: {res}"

    return out
