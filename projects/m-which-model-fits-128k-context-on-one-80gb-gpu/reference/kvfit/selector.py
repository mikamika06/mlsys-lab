import kvfit.memory as mem


def fits_in_gpu(specs, context_len, weights_bytes, bits, gpu_limit=80 * 1024**3):
    kv = mem.kv_cache_bytes(specs, context_len, bits)
    return (weights_bytes + kv) <= gpu_limit
