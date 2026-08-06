from kvcalc.calc import compute_kv_cache_bytes


def find_max_gpu_context(config, weights_bytes, vram_bytes):
    if weights_bytes >= vram_bytes:
        return 0
    available = vram_bytes - weights_bytes
    per_token = compute_kv_cache_bytes(config, 1)
    if per_token <= 0:
        return 0
    return int(available // per_token)
