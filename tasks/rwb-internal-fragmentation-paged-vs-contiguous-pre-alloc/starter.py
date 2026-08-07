import math

def internal_fragmentation(seqlens: list[int], block_size: int, max_len: int) -> tuple[int, int]:
    """Return (paged_waste, contig_waste) as a tuple of two numbers.

    seqlens  – 1-D Python integer array of sequence lengths
    block_size – B, the size of one page/block
    max_len    – L, the maximum sequence length
    """
    raise NotImplementedError('your code here')
