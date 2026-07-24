import numpy as np

def internal_fragmentation(seqlens, block_size, max_len):
    """Return (paged_waste, contig_waste) as a tuple of two numbers.

    seqlens  – 1-D NumPy integer array of sequence lengths
    block_size – B, the size of one page/block
    max_len    – L, the maximum sequence length
    """
    raise NotImplementedError('your code here')
