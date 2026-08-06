from .calc import bytes_per_token

def max_concurrency(config: dict, vram_bytes: int, context_len: int, weights_bytes: int) -> int:
    available_vram = vram_bytes - weights_bytes
    if available_vram <= 0:
        return 0
    bpt = bytes_per_token(config)
    if bpt <= 0:
        return 0
    per_req_vram = bpt * context_len
    return int(available_vram // per_req_vram)
