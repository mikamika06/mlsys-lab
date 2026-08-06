def measure_allocated_blocks(block_table, block_size):
    if not block_table:
        return 0
    unique_blocks = set()
    for seq_blocks in block_table.values():
        for b in seq_blocks:
            if b >= 0:
                unique_blocks.add(b)
    return len(unique_blocks)
