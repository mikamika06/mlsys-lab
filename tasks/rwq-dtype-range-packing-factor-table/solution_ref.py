def dtype_range_packing() -> dict[str, tuple[int, int, int]]:
    """Return symmetric range and packing factor for qint2, qint4, qint8."""
    def info(b):
        q_min = - (1 << (b - 1))
        q_max = (1 << (b - 1)) - 1
        pack_factor = 8 // b
        return (q_min, q_max, pack_factor)

    return {
        "qint2": info(2),
        "qint4": info(4),
        "qint8": info(8)
    }
