def paged_vs_contiguous_concurrency(seqlens, n_blocks, block_size, max_len):
    """
    seqlens: 1-D array/list of positive int request lengths, arrival order.
    n_blocks: total physical block budget N.
    block_size: tokens per block B.
    max_len: worst-case context length L_max reserved by the contiguous
        (slot-based) allocator.

    Returns (max_concurrent_paged, max_concurrent_contig):
      - max_concurrent_paged: greedily admit requests from seqlens in
        order, each costing ceil(length / block_size) blocks, until the
        next request would exceed n_blocks.
      - max_concurrent_contig: n_blocks // ceil(max_len / block_size),
        independent of the actual seqlens.
    """
    raise NotImplementedError('your code here')
