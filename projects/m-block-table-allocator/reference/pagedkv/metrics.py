def compute_fragmentation(allocator, seq_lengths: dict[str, int]) -> dict[str, float]:
    """Computes internal and external fragmentation metrics."""
    total_allocated_slots = 0
    total_used_slots = 0

    for seq_id, seq_len in seq_lengths.items():
        bt = allocator.get_block_table(seq_id)
        allocated_slots = len(bt) * allocator.block_size
        total_allocated_slots += allocated_slots
        total_used_slots += seq_len

    if total_allocated_slots == 0:
        internal_frag = 0.0
    else:
        internal_frag = (total_allocated_slots - total_used_slots) / total_allocated_slots

    free_blocks = allocator.get_num_free_blocks()
    free_slots = free_blocks * allocator.block_size
    total_capacity = allocator.num_blocks * allocator.block_size

    if total_capacity == 0:
        external_frag = 0.0
    else:
        external_frag = free_slots / total_capacity

    return {
        "internal_fragmentation": float(internal_frag),
        "external_fragmentation": float(external_frag),
    }
