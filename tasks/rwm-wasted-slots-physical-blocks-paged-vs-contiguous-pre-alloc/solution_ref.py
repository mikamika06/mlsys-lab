def wasted_slots(lengths, bs):
    """
    Compute the number of blocks and wasted slots for paged and contiguous pre‑allocation.

    Parameters
    ----------
    lengths : list[int]
        Lengths of each sequence.
    bs : int
        Block size (must be > 0).

    Returns
    -------
    dict[str, tuple[int,int]]
        {'paged': (num_blocks_paged, wasted_paged),
         'contiguous': (num_blocks_contig, wasted_contig)}
    """
    if not lengths:
        return {"paged": (0, 0), "contiguous": (0, 0)}

    max_len = max(lengths)
    paged_blocks = sum((l + bs - 1) // bs for l in lengths)
    paged_wasted = paged_blocks * bs - sum(lengths)

    contig_block_per_seq = (max_len + bs - 1) // bs
    contig_blocks = contig_block_per_seq * len(lengths)
    contig_wasted = contig_blocks * bs - sum(lengths)

    return {
        "paged": (paged_blocks, paged_wasted),
        "contiguous": (contig_blocks, contig_wasted)
    }
