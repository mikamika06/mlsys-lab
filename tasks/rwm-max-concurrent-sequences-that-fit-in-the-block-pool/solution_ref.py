import math

def max_concurrent_sequences(lengths, block_size, total_blocks):
    """Return the maximum number of concurrent sequences fitting in the block pool.

    Uses greedy shortest-first: sort by ceil(length/block_size) ascending,
    admit until the pool is exhausted.
    """
    blocks_needed = sorted(
        (l + block_size - 1) // block_size for l in lengths
    )
    used = 0
    count = 0
    for b in blocks_needed:
        if used + b <= total_blocks:
            used += b
            count += 1
        else:
            break
    return count
