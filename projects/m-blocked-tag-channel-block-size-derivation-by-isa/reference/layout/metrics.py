def count_reorders(shape: tuple, layout: str, isa_name: str) -> int:
    n, c, h, w = shape
    base = n * h * w
    if layout == "channels_last":
        if isa_name == "neon":
            return base * max(1, c // 4)
        elif isa_name == "avx2":
            return base * max(1, c // 8)
        return base * c
    return base * c // 2

def overhead_fraction(plain_ops: int, blocked_ops: int) -> float:
    if plain_ops == 0:
        return 0.0
    return float(abs(plain_ops - blocked_ops)) / float(plain_ops + blocked_ops)
