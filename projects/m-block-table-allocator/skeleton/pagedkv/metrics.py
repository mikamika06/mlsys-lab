class BlockAllocator:
    pass


def compute_fragmentation(allocator, seq_lengths: dict[str, int]) -> dict[str, float]:
    """Computes internal and external fragmentation metrics."""
    raise NotImplementedError
