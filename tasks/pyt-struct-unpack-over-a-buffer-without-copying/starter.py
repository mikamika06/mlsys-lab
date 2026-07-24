def parse_records_view(buf: bytearray, n: int, record_size: int = 12) -> list:
    """Zero-copy (id_view, x_view, y_view) memoryview triples over `buf`,
    one per <iff> record, built purely via memoryview slicing/.cast()."""
    raise NotImplementedError('your code here')
