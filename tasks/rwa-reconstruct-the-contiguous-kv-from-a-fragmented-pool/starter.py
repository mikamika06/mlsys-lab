def reconstruct_contiguous_kv(kv_pool: list[list[float]], block_table: list[int], block_size: int, seq_len: int) -> list[list[float]]:
    """Rebuild the logical, contiguous (seq_len, d) KV tensor for one
    sequence out of a shared PAGED physical pool.

    kv_pool     : (num_physical_blocks * block_size, d). This sequence's
                  data lives scattered across the physical blocks named
                  by `block_table`; every other row belongs to other
                  sequences (or is unused) and must be ignored.
    block_table : list of physical block ids for this sequence's LOGICAL
                  blocks 0, 1, 2, ... in order. len(block_table) * block_size
                  >= seq_len. Physical ids need not be contiguous or in
                  logical order.
    block_size  : tokens per physical block.
    seq_len     : number of logical positions to reconstruct.

    Returns a (seq_len, d) array where row `pos` equals
    kv_pool[block_table[pos // block_size] * block_size + pos % block_size].
    """
    raise NotImplementedError('your code here')
