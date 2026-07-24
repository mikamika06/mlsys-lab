def count_transactions(base_addr: int, stride: int, num_threads: int = 32) -> int:
    """Compute the number of 128-byte memory transactions for a warp access.

    Thread i accesses byte address base_addr + i * stride.
    Each address falls into the 128-byte segment indexed by addr // 128.
    Returns the number of distinct segments touched.
    """
    segments = set()
    for i in range(num_threads):
        addr = base_addr + i * stride
        segments.add(addr // 128)
    return len(segments)
