import ref

def check(workdir):
    from pagedkv.cache import BlockCache
    m = {"api_ok": 0.0}
    c = BlockCache(block_size=16, num_blocks=4, head_dim=64)
    b0 = c.allocate()
    if b0 is None or c.free_count() != 3:
        return m
    c.release(b0)
    if c.free_count() != 4:
        return m
    m["api_ok"] = 1.0
    return m
