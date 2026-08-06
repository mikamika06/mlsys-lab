import numpy as np
import math

def internal_fragmentation(seqlens, block_size, max_len):
    """Return (paged_waste, contig_waste) as a tuple of two numbers.

    Paged waste:  sum_i ( ceil(l_i / B) * B - l_i )
    Contig waste: sum_i ( L - l_i )
    """
    seqlens = np.asarray(seqlens, dtype=np.int64)
    paged_waste = 0
    contig_waste = 0
    for l_i in seqlens:
        num_blocks = math.ceil(l_i / float(block_size))
        paged_waste += int(num_blocks * block_size - l_i)
        contig_waste += int(max_len - l_i)
    return paged_waste, contig_waste
