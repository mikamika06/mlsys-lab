def prefix_block_share(sequences: list[list[int]], block_size: int = 4) -> dict:
    """
    Compute block-aligned chained prefix hashes for each sequence, and derive
    the number of physical blocks needed / saved by cross-sequence sharing.

    Returns a dict with keys "block_hashes", "num_physical_blocks", "blocks_saved".
    """
    raise NotImplementedError('your code here')
