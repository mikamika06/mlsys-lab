def compute_slot_mapping(block_tables, block_size):
    """Computes flat physical slot mapping for batch."""
    slots = []
    for table, seq_len in block_tables:
        for i in range(seq_len):
            b = table[i // block_size]
            off = i % block_size
            slots.append(b * block_size + off)
    return slots
