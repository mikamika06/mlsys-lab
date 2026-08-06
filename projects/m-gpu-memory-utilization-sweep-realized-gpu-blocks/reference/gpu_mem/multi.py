def partition_blocks(total_blocks: int, util_a: float, util_b: float) -> tuple:
    sum_util = util_a + util_b
    if sum_util <= 0:
        return (0, 0)
    blocks_a = int(total_blocks * (util_a / sum_util))
    blocks_b = total_blocks - blocks_a
    return (blocks_a, blocks_b)
