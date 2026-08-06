def compute_internal_fragmentation(seq_lengths, block_size):
    """Computes total internal fragmentation in tokens for a sequence distribution."""
    total_frag = 0
    for length in seq_lengths:
        if length > 0:
            remainder = length % block_size
            if remainder != 0:
                total_frag += block_size - remainder
    return total_frag


def derive_optimal_block_size(seq_lengths, candidate_block_sizes):
    """Finds candidate block size that minimizes total internal fragmentation."""
    best_size = None
    min_frag = float("inf")
    
    for size in candidate_block_sizes:
        frag = compute_internal_fragmentation(seq_lengths, size)
        if frag < min_frag:
            min_frag = frag
            best_size = size
    return best_size, min_frag
