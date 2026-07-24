def minimal_block_allocation(positions, block_size):
    blocks = sorted({p // block_size for p in positions})
    offsets = [p % block_size for p in positions]
    return blocks, offsets
