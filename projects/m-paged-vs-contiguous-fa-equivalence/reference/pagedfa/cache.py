import numpy as np


def allocate_paged_cache(k_contig, v_contig, seq_lens, block_size, total_blocks):
    """Allocate physical block pools and populate logical block tables."""
    batch_size, max_len, num_heads, head_dim = k_contig.shape
    k_pool = np.zeros((total_blocks, block_size, num_heads, head_dim), dtype=k_contig.dtype)
    v_pool = np.zeros((total_blocks, block_size, num_heads, head_dim), dtype=v_contig.dtype)

    max_blocks_per_seq = (max_len + block_size - 1) // block_size
    block_tables = np.full((batch_size, max_blocks_per_seq), -1, dtype=np.int32)

    free_blocks = list(range(total_blocks))
    for b in range(batch_size):
        slen = seq_lens[b]
        needed_blocks = (slen + block_size - 1) // block_size
        for blk_idx in range(needed_blocks):
            phy_blk = free_blocks.pop(0)
            block_tables[b, blk_idx] = phy_blk
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, slen)
            valid_len = end_pos - start_pos
            k_pool[phy_blk, :valid_len] = k_contig[b, start_pos:end_pos]
            v_pool[phy_blk, :valid_len] = v_contig[b, start_pos:end_pos]

    return k_pool, v_pool, block_tables
