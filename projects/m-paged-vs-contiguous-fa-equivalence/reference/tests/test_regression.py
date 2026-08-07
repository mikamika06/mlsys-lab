import numpy as np
from paged_fa.utils import reconstruct_contiguous
from paged_fa.attention import standard_attention, paged_attention

def test_paged_attention_matches_standard():
    np.random.seed(42)
    batch_size = 2
    num_heads = 2
    head_dim = 16
    block_size = 4
    num_blocks = 16

    q = np.random.randn(batch_size, num_heads, head_dim)
    k_cache = np.random.randn(num_blocks, block_size, num_heads, head_dim)
    v_cache = np.random.randn(num_blocks, block_size, num_heads, head_dim)

    context_lens = np.array([5, 7])
    block_tables = np.array([
        [2, 5],
        [8, 1]
    ])

    k_contig, v_contig = reconstruct_contiguous(k_cache, v_cache, block_tables, context_lens)

    out_std = standard_attention(q, k_contig, v_contig, context_lens)
    out_paged = paged_attention(q, k_cache, v_cache, block_tables, context_lens)

    assert np.max(np.abs(out_std - out_paged)) < 1e-5
