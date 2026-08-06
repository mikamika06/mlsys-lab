def measure_allocated_blocks(block_tables, block_size):
    unique_blocks = set()
    for table in block_tables:
        for block_id in table:
            if block_id >= 0:
                unique_blocks.add(block_id)
    return len(unique_blocks)
