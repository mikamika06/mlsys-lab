import numpy as np

def internal_fragmentation(seqlens, block_size, max_len):
    """Return (paged_waste, contig_waste) as a tuple of two numbers.

    Paged waste:  sum_i ( ceil(l_i / B) * B - l_i )
    Contig waste: sum_i ( L - l_i )
    """
    seqlens = np.asarray(seqlens, dtype=np.int64)
    # Number of full blocks per sequence (ceiling division)
    num_blocks = np.ceil(seqlens / np.float64(block_size)).astype(np.int64)
    paged_waste = int(np.sum(num_blocks * block_size - seqlens))
    contig_waste = int(np.sum(max_len - seqlens))
    return paged_waste, contig_waste
