def blocks_per_pool(alignment: int, pool_size: int, pool_header_size: int,
                    max_block_size: int) -> list[int]:
    """Compute blocks per pool for each pymalloc size class."""
    raise NotImplementedError("your code here")
