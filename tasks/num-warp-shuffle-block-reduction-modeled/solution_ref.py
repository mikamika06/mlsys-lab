import math


def block_reduce(values, block_size, warp_size=4):
    if block_size != len(values):
        raise ValueError("block_size must equal input length")

    warps = math.ceil(block_size / warp_size)
    stages = int(math.log2(warp_size))

    global_reads = block_size
    shared_writes = warps
    shuffle_accesses = warps * warp_size * stages

    total_accesses = global_reads + shared_writes + shuffle_accesses

    if warps > 1:
        shared_reads = warps
        final_shuffle_accesses = warp_size * stages
        total_accesses += shared_reads + final_shuffle_accesses

    return float(sum(values)), int(total_accesses)
