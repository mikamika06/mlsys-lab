def hash_block(token_ids, parent_hash=None):
    """Computes deterministic hash for a block of token IDs given parent block hash."""
    raise NotImplementedError


def compute_prefix_hashes(token_ids, block_size):
    """Computes chain of block hashes for token sequence."""
    raise NotImplementedError
