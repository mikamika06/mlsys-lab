def compute_budget(seq_lens, block_size, num_layers):
    total = 0
    for length in seq_lens:
        blocks_per_layer = (length + block_size - 1) // block_size
        total += blocks_per_layer * num_layers
    return total
