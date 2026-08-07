import math

def mxfp4_block_exponent(x: list[float], block_size: int=32) -> list[int]:
    """Compute the MX (microscaling) E8M0 power-of-two block exponent.

    x: 1-D float array, length an exact multiple of block_size.
    Returns an int array of shape (len(x) // block_size,): for each block,
    the exponent e = floor(log2(amax / 6)) where amax = max(abs(block))
    (special case: amax == 0 -> exponent 0).
    """
    raise NotImplementedError('your code here')
