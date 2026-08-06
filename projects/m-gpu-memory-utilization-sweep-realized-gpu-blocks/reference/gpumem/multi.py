def partition_gpu_blocks(total_blocks: int, util_a: float, util_b: float) -> tuple:
    total_util = util_a + util_b
    if total_util <= 0:
        return (0, 0)
    blocks_a = int(total_blocks * (util_a / total_util))
    blocks_b = total_blocks - blocks_a
    return (blocks_a, blocks_b)
