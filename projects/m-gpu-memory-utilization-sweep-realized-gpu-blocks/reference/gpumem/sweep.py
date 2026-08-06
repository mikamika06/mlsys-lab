def realized_blocks(total_vram: int, weight_bytes: int, overhead_bytes: int, block_bytes: int, util: float) -> int:
    avail = int(total_vram * util)
    usable = avail - weight_bytes - overhead_bytes
    if usable <= 0:
        return 0
    return usable // block_bytes
