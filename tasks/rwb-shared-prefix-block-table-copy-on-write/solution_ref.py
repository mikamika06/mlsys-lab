def _lcp_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _ceil_div(x, y):
    return -(-x // y)


def build_shared_prefix_block_tables(tokens_a: list, tokens_b: list, block_size: int):
    """
    tokens_a, tokens_b: two token-id sequences (possibly sharing a
        common prefix). block_size: B, the PagedAttention block size.

    Determine the shared prefix length P = length of the longest common
    prefix of tokens_a and tokens_b. Build a block table for each
    sequence: ceil(len(seq) / B) logical block slots, each holding a
    physical block id.

    The first ceil(P / B) block-table entries of BOTH sequences point to
    the SAME physical block ids (aliased -- these blocks are never
    duplicated, even if P isn't a multiple of B: that last, partially
    filled prefix block still counts as one shared block). Every
    remaining ("tail") block-table entry, for EACH sequence
    independently, gets its own freshly allocated, globally unique
    physical block id -- tail blocks are never shared between the two
    sequences and never reuse a shared block's id.

    Physical block ids are assigned starting from 0, in increasing
    order: first the shared prefix blocks, then sequence A's tail
    blocks, then sequence B's tail blocks.

    Return (block_table_a, block_table_b, num_physical_blocks).
    """
    P = _lcp_len(tokens_a, tokens_b)
    n_shared = _ceil_div(P, block_size)
    n_blocks_a = _ceil_div(len(tokens_a), block_size)
    n_blocks_b = _ceil_div(len(tokens_b), block_size)
    tail_a = n_blocks_a - n_shared
    tail_b = n_blocks_b - n_shared

    shared_ids = list(range(n_shared))
    nxt = n_shared
    tail_a_ids = list(range(nxt, nxt + tail_a))
    nxt += tail_a
    tail_b_ids = list(range(nxt, nxt + tail_b))
    nxt += tail_b

    block_table_a = shared_ids + tail_a_ids
    block_table_b = shared_ids + tail_b_ids
    return block_table_a, block_table_b, nxt
