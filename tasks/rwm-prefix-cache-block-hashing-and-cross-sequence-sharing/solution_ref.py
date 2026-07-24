def _block_hashes(seq: list[int], block_size: int) -> list[int]:
    n_full = len(seq) // block_size
    parent = None
    hashes = []
    for i in range(n_full):
        block = tuple(seq[i * block_size:(i + 1) * block_size])
        h = hash((parent, block))
        hashes.append(h)
        parent = h
    return hashes


def prefix_block_share(sequences: list[list[int]], block_size: int = 4) -> dict:
    """
    Compute block-aligned chained prefix hashes for each sequence, and derive
    the number of physical blocks needed / saved by cross-sequence sharing.
    """
    block_hashes = [_block_hashes(seq, block_size) for seq in sequences]

    unique = set()
    for hs in block_hashes:
        unique.update(hs)

    total_naive = sum(len(hs) for hs in block_hashes)
    num_physical_blocks = len(unique)
    blocks_saved = total_naive - num_physical_blocks

    return {
        "block_hashes": block_hashes,
        "num_physical_blocks": num_physical_blocks,
        "blocks_saved": blocks_saved,
    }
