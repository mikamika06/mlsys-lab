def paged_vs_contiguous_concurrency(seqlens, n_blocks, block_size, max_len):
    used = 0
    paged = 0
    for length in seqlens:
        blocks = -(-int(length) // block_size)  # ceil division
        if used + blocks > n_blocks:
            break
        used += blocks
        paged += 1

    contig_blocks_per_req = -(-int(max_len) // block_size)
    contig = n_blocks // contig_blocks_per_req

    return paged, contig
