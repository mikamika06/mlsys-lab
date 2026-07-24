def parse_records_view(buf: bytearray, n: int, record_size: int = 12) -> list:
    """Zero-copy (id_view, x_view, y_view) memoryview triples over `buf`,
    one per <iff> record, built purely via memoryview slicing/.cast()."""
    mv = memoryview(buf)
    out = []
    for i in range(n):
        rec = mv[i * record_size:(i + 1) * record_size]
        id_view = rec[0:4].cast('i')
        x_view = rec[4:8].cast('f')
        y_view = rec[8:12].cast('f')
        out.append((id_view, x_view, y_view))
    return out
