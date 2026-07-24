def arithmetic_intensity(flops: int,
                         read_bytes: int,
                         write_bytes: int = 0) -> float:
    total = read_bytes + write_bytes
    if total == 0:
        return float('inf')
    ai = flops / total
    return round(ai, 6)
