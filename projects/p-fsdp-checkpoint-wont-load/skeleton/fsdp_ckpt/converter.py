def consolidate(aligned, metadata):
    """Reconstruct the full parameter arrays."""
    raise NotImplementedError


def shard_checkpoint(consolidated, num_ranks):
    """Split parameters for a specific number of ranks."""
    raise NotImplementedError
