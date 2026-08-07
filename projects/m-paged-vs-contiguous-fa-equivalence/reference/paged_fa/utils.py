import numpy as np

def reconstruct_contiguous(k_cache, v_cache, block_tables, context_lens):
    batch_size = len(context_lens)
    max_seq_len = max(context_lens)
    _, block_size, num_heads, head_dim = k_cache.shape

    k_contig = np.zeros((batch_size, max_seq_len, num_heads, head_dim))
    v_contig = np.zeros((batch_size, max_seq_len, num_heads, head_dim))

    for b in range(batch_size):
        seq_len = context_lens[b]
        for i in range(seq_len):
            logical_block = i // block_size
            offset = i % block_size
            physical_block = block_tables[b][logical_block]
            k_contig[b, i] = k_cache[physical_block, offset]
            v_contig[b, i] = v_cache[physical_block, offset]

    return k_contig, v_contig
