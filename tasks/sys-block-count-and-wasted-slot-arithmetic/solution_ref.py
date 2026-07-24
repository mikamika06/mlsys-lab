def compute_block_and_waste(block_size, seq_lengths):
    blocks = []
    wasted = []
    for l in seq_lengths:
        b = (l + block_size - 1) // block_size
        w = b * block_size - l
        blocks.append(b)
        wasted.append(w)
    return blocks, wasted
