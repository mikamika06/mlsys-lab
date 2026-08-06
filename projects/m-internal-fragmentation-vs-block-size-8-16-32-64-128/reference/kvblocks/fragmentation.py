def compute_fragmentation(seq_lens, block_sizes):
    res = {}
    for bs in block_sizes:
        total_allocated = 0
        total_used = sum(seq_lens)
        for length in seq_lens:
            num_blocks = (length + bs - 1) // bs
            total_allocated += num_blocks * bs
        wasted = total_allocated - total_used
        frag = wasted / total_allocated if total_allocated > 0 else 0.0
        res[bs] = frag
    return res
