def fp8_capacity(total_vram: int, weight_bytes: int, overhead_bytes: int, block_bytes_fp16: int, util: float) -> int:
    avail = int(total_vram * util)
    usable = avail - weight_bytes - overhead_bytes
    if usable <= 0:
        return 0
    block_bytes_fp8 = block_bytes_fp16 // 2
    return usable // block_bytes_fp8
