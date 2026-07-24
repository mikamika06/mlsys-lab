def process(sizes: list, make_buffer) -> float:
    """Stream through sizes, one large object alive at a time.

    Reassigning `buf` each iteration drops the previous object's
    reference the moment the new one is bound, so CPython's real
    refcounting frees it immediately (peak live count stays at 2, not N).
    """
    total = 0.0
    for s in sizes:
        buf = make_buffer(s)
        total += buf.checksum()
    return total
