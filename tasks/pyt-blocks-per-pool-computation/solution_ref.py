def blocks_per_pool(alignment: int, pool_size: int, pool_header_size: int,
                    max_block_size: int) -> list[int]:
    available = pool_size - pool_header_size
    result = []
    s = alignment
    while s <= max_block_size:
        result.append(available // s)
        s += alignment
    return result
