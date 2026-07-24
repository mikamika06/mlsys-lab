def choose_padding(width: int) -> int:
    """
    Return the smallest non-negative pad such that a warp doing a
    column-stride shared-memory access with stride (width + pad) is
    conflict-free under the 32-bank model (i.e. width + pad is odd).
    """
    raise NotImplementedError('your code here')
