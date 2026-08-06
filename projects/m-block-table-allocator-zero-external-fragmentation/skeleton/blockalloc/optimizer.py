def compute_internal_fragmentation(seq_lengths, block_size):
    """Computes total internal fragmentation in tokens for a sequence distribution."""
    raise NotImplementedError


def derive_optimal_block_size(seq_lengths, candidate_block_sizes):
    """Finds candidate block size that minimizes total internal fragmentation."""
    raise NotImplementedError
